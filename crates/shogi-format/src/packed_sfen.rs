//! PackedSfen / PackedSfenValue デコーダ
//!
//! 40-byte 圧縮局面 + score / move / WDL の教師データ形式を読み込むための
//! モジュール。PackedSfenValue (40 bytes) から局面を復元し、特徴量計算に
//! 使用する。
//!
//! Huffman 表 (`HUFFMAN_TABLE`) と「成りbit + 先後bit」の bit layout は本 doc
//! 内 / 関数 doc に直接記述している。駒箱 (hand 駒種外の駒) のフラグ判定にも
//! 対応 (`decode_hand_piece` 参照)。
//!
//! 駒種 Huffman 復号は `HUFFMAN_BOARD_LUT` (盤上、64 entry) /
//! `HUFFMAN_HAND_LUT` (手駒、32 entry) の `peek_bits` 1 回 → table 1 lookup で
//! `(piece_idx, code_bits)` を取得する形に集約している。LUT は `HUFFMAN_TABLE`
//! の prefix code から `const fn` で生成しており、テーブル本体 (LSB-first
//! pattern + bit 長) を改変すれば LUT も自動で追随する。

use super::types::{BOARD_PIECE_TYPES, Color, Hand, Piece, PieceType, Square};

// =============================================================================
// Huffman 符号テーブル
// =============================================================================

/// Huffman 符号テーブル (PackedSfen の駒種 encoding に対応)
///
/// インデックス: 0=NO_PIECE, 1=PAWN, 2=LANCE, 3=KNIGHT, 4=SILVER, 5=BISHOP, 6=ROOK, 7=GOLD
const HUFFMAN_TABLE: [(u32, u8); 8] = [
    (0x00, 1), // NO_PIECE: 0
    (0x01, 2), // PAWN:     01
    (0x03, 4), // LANCE:    0011
    (0x0b, 4), // KNIGHT:   1011
    (0x07, 4), // SILVER:   0111
    (0x1f, 6), // BISHOP:   011111
    (0x3f, 6), // ROOK:     111111
    (0x0f, 5), // GOLD:     01111
];

/// 駒種インデックス（Huffman テーブル用）。`HUFFMAN_TABLE` の row index と
/// `HuffmanEntry::piece_idx` の両方で同じ値を使う。
const HUFFMAN_NONE: u8 = 0;
const HUFFMAN_PAWN: u8 = 1;
const HUFFMAN_LANCE: u8 = 2;
const HUFFMAN_KNIGHT: u8 = 3;
const HUFFMAN_SILVER: u8 = 4;
const HUFFMAN_BISHOP: u8 = 5;
const HUFFMAN_ROOK: u8 = 6;
const HUFFMAN_GOLD: u8 = 7;

// =============================================================================
// Fast-Huffman lookup table
// =============================================================================

/// 1 駒の Huffman 符号 1 entry。
///
/// `code_bits` は符号自体の bit 長 (`HUFFMAN_TABLE` の `len`)。
/// 成り flag / 先後 flag bit はここには含まない (caller 側で別途読む)。
#[derive(Clone, Copy)]
struct HuffmanEntry {
    piece_idx: u8,
    code_bits: u8,
}

/// 盤上駒 LUT 引きで peek する bit 数 (= `HUFFMAN_TABLE` 中の最大 `len`)。
const HUFFMAN_BOARD_PEEK_BITS: u8 = 6;
/// 手駒 LUT 引きで peek する bit 数 (= board 最大 - 1、bit0 が省略されるため)。
const HUFFMAN_HAND_PEEK_BITS: u8 = 5;

/// `HUFFMAN_TABLE` の prefix code から `peek` 値 → entry の lookup table を
/// 生成する。
///
/// `hand` が true のときは手駒 encoding (盤上符号から bit0 を省略 =
/// `pattern >> 1`、bit 長 -1) を使う。`hand=true` では NO_PIECE は手駒符号化に
/// 出現しないので除外する。
///
/// LSB-first 解釈なので「peek 値の下位 `code_bits` bit が `pattern` に一致」
/// する最初の entry を返す。`HUFFMAN_TABLE` は prefix code (どの符号も他の符号
/// の prefix でない) なので一致は高々 1 通りに決まる。
const fn build_huffman_lut<const N: usize>(hand: bool) -> [HuffmanEntry; N] {
    let mut lut = [HuffmanEntry {
        piece_idx: HUFFMAN_NONE,
        code_bits: 0,
    }; N];
    let mut v: u32 = 0;
    while (v as usize) < N {
        let mut matched = false;
        let mut i = 0;
        while i < HUFFMAN_TABLE.len() {
            if !(hand && i == 0) {
                let (mut pattern, mut len) = HUFFMAN_TABLE[i];
                if hand {
                    pattern >>= 1;
                    len -= 1;
                }
                let mask: u32 = if len >= 32 {
                    u32::MAX
                } else {
                    (1u32 << len) - 1
                };
                if (v & mask) == pattern {
                    lut[v as usize] = HuffmanEntry {
                        piece_idx: i as u8,
                        code_bits: len,
                    };
                    matched = true;
                    break;
                }
            }
            i += 1;
        }
        // `HUFFMAN_TABLE` の prefix code は board (peek 6 bit) / hand (peek 5 bit)
        // 共に全 peek 値を網羅する設計 (各 v に一意に一致)。網羅性が崩れたら
        // const-eval 中に落として LUT 生成段階で error にする。
        if !matched {
            panic!("HUFFMAN_TABLE prefix code does not cover all peek values");
        }
        v += 1;
    }
    lut
}

const HUFFMAN_BOARD_LUT: [HuffmanEntry; 1 << HUFFMAN_BOARD_PEEK_BITS] = build_huffman_lut(false);
const HUFFMAN_HAND_LUT: [HuffmanEntry; 1 << HUFFMAN_HAND_PEEK_BITS] = build_huffman_lut(true);

// =============================================================================
// ビットストリーム
// =============================================================================

/// LSB-first ビットストリーム
pub struct BitStream<'a> {
    data: &'a [u8],
    bit_cursor: usize,
    /// ビット単位の上限（これ以上は読み出し不可）
    bit_limit: usize,
}

impl<'a> BitStream<'a> {
    /// 新しいビットストリームを作成
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            bit_cursor: 0,
            bit_limit: data.len() * 8,
        }
    }

    /// 現在のカーソル位置（ビット単位）
    #[inline]
    pub fn cursor(&self) -> usize {
        self.bit_cursor
    }

    /// 読み出し可能かどうか
    #[inline]
    pub fn can_read(&self) -> bool {
        self.bit_cursor < self.bit_limit
    }

    /// 1ビット読み出し（境界チェック付き）
    ///
    /// 境界を超えた場合は false を返す（安全なデフォルト）
    #[inline]
    pub fn read_bit(&mut self) -> bool {
        if self.bit_cursor >= self.bit_limit {
            return false;
        }
        let byte_pos = self.bit_cursor / 8;
        let bit_pos = self.bit_cursor % 8;
        let bit = (self.data[byte_pos] >> bit_pos) & 1;
        self.bit_cursor += 1;
        bit != 0
    }

    /// nビット読み出し（最大32ビット）
    #[inline]
    pub fn read_bits(&mut self, n: u8) -> u32 {
        let mut result = 0u32;
        for i in 0..n {
            if self.read_bit() {
                result |= 1 << i;
            }
        }
        result
    }

    /// カーソルを進めずに n bit を先読みする (LSB-first, n ≤ 25)。
    ///
    /// 4-byte unaligned load + shift / mask の 1 pass。`bit_cursor` が
    /// `bit_limit` を越えていても panic しない: slice 起点を `data.len()` に
    /// clamp してから copy するので `&data[i..i]` で i > len になる経路は無く、
    /// 末尾を越えた byte は 0 で埋まる。0 padding は board LUT では NO_PIECE、
    /// hand LUT では PAWN code (盤上 PAWN の `01` から bit0 を省いた `0`) に
    /// 該当するが、`from_packed_sfen` は board を 79 駒固定 / hand を
    /// `cursor() < 256` で終端制御するため、末尾を越えた peek の戻り値が
    /// 偶発的に新しい駒として消費されることは無い。
    /// `n > 25` のときは bit_cursor の byte 内 offset (最大 7) と合わせて 32 bit
    /// を越えるので debug_assert で落とす。
    #[inline]
    pub fn peek_bits(&self, n: u8) -> u32 {
        debug_assert!(n <= 25, "peek_bits supports up to 25 bits (32 - 7 = 25)");
        let bit_off = (self.bit_cursor % 8) as u32;
        // bit_cursor が bit_limit を越え (advance で進めた後等)、(bit_cursor / 8)
        // が data.len() を越えても panic しないよう clamp する。clamp 後の
        // [start..start+copy_len] は常に `data` の有効 sub-slice。
        let start = (self.bit_cursor / 8).min(self.data.len());

        let mut buf = [0u8; 4];
        let copy_len = (self.data.len() - start).min(buf.len());
        buf[..copy_len].copy_from_slice(&self.data[start..start + copy_len]);
        let word = u32::from_le_bytes(buf);

        let mask: u32 = if n >= 32 { u32::MAX } else { (1u32 << n) - 1 };
        (word >> bit_off) & mask
    }

    /// カーソルを n bit 進める。`peek_bits` で取り出した bit の消費に使う。
    ///
    /// `read_bit` と違い OOB clamp はしない (bit_cursor は bit_limit を越え得る)。
    /// 越えた後の `read_bit` は false を返すので decode loop 側で問題ない。
    #[inline]
    pub fn advance(&mut self, n: usize) {
        self.bit_cursor += n;
    }
}

// =============================================================================
// PackedSfen (32バイト)
// =============================================================================

/// Huffman符号化された局面データ (32バイト)
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PackedSfen {
    pub data: [u8; 32],
}

impl PackedSfen {
    /// バイト配列への参照を取得
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.data
    }

    /// バイト配列への可変参照を取得
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        &mut self.data
    }
}

// =============================================================================
// PackedSfenValue (40バイト)
// =============================================================================

/// 教師データ 1 局面 (40 バイト固定、PackedSfen + score/move/ply/WDL)。
///
/// これが `SparseInputType::RequiredDataType` として使用される。
///
/// メモリレイアウト:
/// - bytes 0-31:  PackedSfen (32 bytes)
/// - bytes 32-33: score (i16, little-endian)
/// - bytes 34-35: move16 (u16, little-endian)
/// - bytes 36-37: game_ply (u16, little-endian)
/// - byte 38:     game_result (i8)
/// - byte 39:     padding (u8)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PackedSfenValue {
    data: [u8; 40],
}

impl Default for PackedSfenValue {
    fn default() -> Self {
        Self { data: [0u8; 40] }
    }
}

// SAFETY: `PackedSfenValue` は `[u8; 40]` 1 個だけを持つ POD (内部可変性・生ポインタ無し)
// なので thread 間で安全に送受信・共有できる。
unsafe impl Send for PackedSfenValue {}
unsafe impl Sync for PackedSfenValue {}

impl PackedSfenValue {
    /// PackedSfen 部分 (先頭 32 bytes) への参照を取得。
    pub fn sfen(&self) -> &PackedSfen {
        // SAFETY:
        // - `PackedSfen` は `#[repr(C)]` で唯一の field が `[u8; 32]` → サイズ 32 bytes、
        //   align = 1。`PackedSfenValue` (`#[repr(C)]`、field は `[u8; 40]`) の先頭 32 bytes が
        //   ちょうど `PackedSfen` の中身に対応する (PSV wire 形式の bytes 0-31)。
        // - `self.data.as_ptr()` は offset 0 を指し、`PackedSfen` の align=1 を満たす。
        // - `[u8; N]` には不正ビットパターンが無いので、任意のバイト列が valid な `PackedSfen`。
        // - 返す参照のライフタイムは `&self` に縛られる (data は `self` が所有)。
        unsafe { &*(self.data.as_ptr() as *const PackedSfen) }
    }

    /// 評価値を取得（手番側視点）
    pub fn score(&self) -> i16 {
        i16::from_le_bytes([self.data[32], self.data[33]])
    }

    /// 評価値を上書きする（手番側視点、wire 形式 bytes 32-33、little-endian）
    pub fn set_score(&mut self, score: i16) {
        self.data[32..34].copy_from_slice(&score.to_le_bytes());
    }

    /// 指し手を取得
    pub fn move16(&self) -> u16 {
        u16::from_le_bytes([self.data[34], self.data[35]])
    }

    /// 手数を取得
    pub fn game_ply(&self) -> u16 {
        u16::from_le_bytes([self.data[36], self.data[37]])
    }

    /// 勝敗結果を取得
    /// 1=手番側の勝ち, 0=引き分け, -1=手番側の負け
    pub fn game_result(&self) -> i8 {
        self.data[38] as i8
    }

    /// バイトスライスへの参照を取得
    pub fn as_bytes(&self) -> &[u8; 40] {
        &self.data
    }

    /// バイトスライスへの可変参照を取得
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 40] {
        &mut self.data
    }

    /// 局面をデコードして ShogiBoard を返す
    pub fn decode(&self) -> ShogiBoard {
        ShogiBoard::from_packed_sfen(self)
    }
}

// =============================================================================
// ShogiBoard - デコード済み局面
// =============================================================================

/// デコード済みの将棋局面
///
/// PackedSfenValue からデコードした結果を保持。
/// `map_features` で使用する。
#[derive(Clone)]
pub struct ShogiBoard {
    /// 盤面 (81マス)
    pub board: [Piece; 81],
    /// 先手の持ち駒
    pub black_hand: Hand,
    /// 後手の持ち駒
    pub white_hand: Hand,
    /// 手番
    pub side_to_move: Color,
    /// 先手玉の位置
    pub black_king_sq: Square,
    /// 後手玉の位置
    pub white_king_sq: Square,
    /// 評価値
    pub score: i16,
    /// 勝敗結果
    pub result: i8,
    /// 手数
    pub ply: u16,
}

impl Default for ShogiBoard {
    fn default() -> Self {
        Self {
            board: [Piece::NONE; 81],
            black_hand: Hand::EMPTY,
            white_hand: Hand::EMPTY,
            side_to_move: Color::Black,
            black_king_sq: Square::NONE,
            white_king_sq: Square::NONE,
            score: 0,
            result: 0,
            ply: 0,
        }
    }
}

impl ShogiBoard {
    /// PackedSfenValue からデコード
    pub fn from_packed_sfen(psv: &PackedSfenValue) -> Self {
        let mut board = ShogiBoard {
            score: psv.score(),
            result: psv.game_result(),
            ply: psv.game_ply(),
            ..Default::default()
        };

        let mut stream = BitStream::new(&psv.sfen().data);

        // 1. 手番 (1 bit)
        board.side_to_move = if stream.read_bit() {
            Color::White
        } else {
            Color::Black
        };

        // 2. 玉の位置 (7 bit × 2)
        let black_king_idx = stream.read_bits(7) as u8;
        let white_king_idx = stream.read_bits(7) as u8;

        board.black_king_sq = Square(black_king_idx);
        board.white_king_sq = Square(white_king_idx);

        // 玉を盤面に配置
        if black_king_idx < 81 {
            board.board[black_king_idx as usize] = Piece::new(Color::Black, PieceType::King);
        }
        if white_king_idx < 81 {
            board.board[white_king_idx as usize] = Piece::new(Color::White, PieceType::King);
        }

        // 3. 盤上の駒 (Huffman符号)
        for sq_idx in 0..81u8 {
            // 玉位置はスキップ
            if sq_idx == black_king_idx || sq_idx == white_king_idx {
                continue;
            }

            let piece = decode_board_piece(&mut stream);
            board.board[sq_idx as usize] = piece;
        }

        // 4. 持ち駒・駒箱 (Huffman符号、256bitまで)
        while stream.cursor() < 256 {
            let (piece, is_piecebox) = decode_hand_piece(&mut stream);

            // 駒箱の駒は無視（駒落ち対応）
            if is_piecebox {
                continue;
            }

            // 持ち駒に追加
            let pt = piece.piece_type;
            match piece.color {
                Color::Black => board.black_hand.add(pt, 1),
                Color::White => board.white_hand.add(pt, 1),
            }
        }

        board
    }

    /// 指定マスの駒を取得
    #[inline]
    pub fn piece_on(&self, sq: Square) -> Piece {
        self.board[sq.index()]
    }

    /// 指定色の玉位置を取得
    #[inline]
    pub fn king_square(&self, color: Color) -> Square {
        match color {
            Color::Black => self.black_king_sq,
            Color::White => self.white_king_sq,
        }
    }

    /// 指定色の持ち駒を取得
    #[inline]
    pub fn hand(&self, color: Color) -> &Hand {
        match color {
            Color::Black => &self.black_hand,
            Color::White => &self.white_hand,
        }
    }

    /// 盤上の指定色・駒種の駒を列挙
    pub fn pieces(&self, color: Color, pt: PieceType) -> impl Iterator<Item = Square> + '_ {
        self.board
            .iter()
            .enumerate()
            .filter(move |(_, p)| p.color == color && p.piece_type == pt)
            .map(|(i, _)| Square(i as u8))
    }

    /// 盤上の駒 (玉以外) を **1 回の board 走査** で `(piece_type, color, ascending
    /// square)` 順に `f(piece, square)` へ供給する。
    ///
    /// 内部では board を 81 マス 1-pass で `(piece_type, color)` の bucket
    /// (固定 stack 配列) に集め、`BOARD_PIECE_TYPES` の順 × `Color::Black/White`
    /// の順で各 bucket を ascending square で iterate する。各 bucket 内の square
    /// 順序は board.iter() の order (= ascending) と一致するため、`pieces(color, pt)`
    /// を 26 通り loop で呼ぶパターンと emit する index 列は byte-identical
    /// (合法局面 / 駒種数を超えた breakage も含めて完全等価、後述)。
    #[inline]
    pub fn for_each_board_piece<F: FnMut(Piece, Square)>(&self, mut f: F) {
        // PieceType discriminant は `#[repr(u8)]` で 0..=14、Color は同 0..=1。
        // bucket index = (piece_type as usize) * 2 + (color as usize)。
        // MAX_PER_PC は board のマス数 = 81 (片色 1 piece type で全マス占有という
        // 物理上の絶対上限) に揃える: `ShogiBoard` は `pub` で任意配置を構成可、
        // 合法局面の自然上限 (歩 9 / と金 18 / 他 ≤ 4 等) を越える ad-hoc 局面
        // (破損 PSV や test fixture) でも emit を欠落させないため。stack ~2.5 KB。
        const PT_VARIANTS: usize = 15;
        const COLORS: usize = 2;
        const MAX_PER_PC: usize = 81;
        const BUCKETS: usize = PT_VARIANTS * COLORS;

        let mut counts = [0u8; BUCKETS];
        let mut squares = [Square::NONE; BUCKETS * MAX_PER_PC];

        for (i, &p) in self.board.iter().enumerate() {
            // None / King は board feature の対象外。
            if matches!(p.piece_type, PieceType::None | PieceType::King) {
                continue;
            }
            let bucket = (p.piece_type as usize) * COLORS + (p.color as usize);
            let n = counts[bucket] as usize;
            // MAX_PER_PC = 81 = board のマス数。`for (i, ..) in board.iter()` で
            // 1 マス 1 push なので bucket 当たり最大 81 (ad-hoc board で全マスを
            // 同 piece で埋めた場合の上限)、超過は型レベルで起こらない defensive
            // 配置。
            debug_assert!(n < MAX_PER_PC);
            squares[bucket * MAX_PER_PC + n] = Square(i as u8);
            counts[bucket] = (n + 1) as u8;
        }

        for &pt in &BOARD_PIECE_TYPES {
            for color in [Color::Black, Color::White] {
                let bucket = (pt as usize) * COLORS + (color as usize);
                let n = counts[bucket] as usize;
                let row_start = bucket * MAX_PER_PC;
                let piece = Piece::new(color, pt);
                for k in 0..n {
                    f(piece, squares[row_start + k]);
                }
            }
        }
    }
}

// =============================================================================
// Huffman デコード
// =============================================================================

/// Huffman 符号インデックスから PieceType に変換
fn huffman_index_to_piece_type(idx: u8) -> PieceType {
    match idx {
        HUFFMAN_PAWN => PieceType::Pawn,
        HUFFMAN_LANCE => PieceType::Lance,
        HUFFMAN_KNIGHT => PieceType::Knight,
        HUFFMAN_SILVER => PieceType::Silver,
        HUFFMAN_BISHOP => PieceType::Bishop,
        HUFFMAN_ROOK => PieceType::Rook,
        HUFFMAN_GOLD => PieceType::Gold,
        _ => PieceType::None,
    }
}

/// 盤上の駒をデコード
///
/// 形式: Huffman 符号 + 成り bit (金以外) + 先後 bit。
fn decode_board_piece(stream: &mut BitStream) -> Piece {
    let entry = HUFFMAN_BOARD_LUT[stream.peek_bits(HUFFMAN_BOARD_PEEK_BITS) as usize];
    stream.advance(entry.code_bits as usize);

    if entry.piece_idx == HUFFMAN_NONE {
        return Piece::NONE;
    }

    let base_pt = huffman_index_to_piece_type(entry.piece_idx);

    // 成りフラグ（金以外）
    let promoted = if entry.piece_idx != HUFFMAN_GOLD {
        stream.read_bit()
    } else {
        false
    };

    // 先後フラグ
    let color = if stream.read_bit() {
        Color::White
    } else {
        Color::Black
    };

    let pt = if promoted { base_pt.promote() } else { base_pt };
    Piece::new(color, pt)
}

/// 手駒をデコード
///
/// 形式: Huffman 符号 (bit0 を省略) + 成り bit (金以外、駒箱判定用) + 先後 bit。
///
/// 戻り値: (駒, 駒箱フラグ)
/// - 駒箱フラグが true の場合、その駒は持ち駒ではなく駒箱の駒。
fn decode_hand_piece(stream: &mut BitStream) -> (Piece, bool) {
    let entry = HUFFMAN_HAND_LUT[stream.peek_bits(HUFFMAN_HAND_PEEK_BITS) as usize];
    stream.advance(entry.code_bits as usize);

    let base_pt = huffman_index_to_piece_type(entry.piece_idx);

    // 成りフラグ（金以外）。これが true なら駒箱の駒。
    let is_piecebox = if entry.piece_idx != HUFFMAN_GOLD {
        stream.read_bit()
    } else {
        false
    };

    // 先後フラグ
    let color = if stream.read_bit() {
        Color::White
    } else {
        Color::Black
    };

    (Piece::new(color, base_pt), is_piecebox)
}

// =============================================================================
// 対局結果 (inherent GameResult アクセサ)
// =============================================================================

impl PackedSfenValue {
    /// 対局結果 (raw i8 game_result から GameResult enum へ)。
    pub fn result(&self) -> crate::GameResult {
        match self.game_result() {
            r if r > 0 => crate::GameResult::Win,
            r if r < 0 => crate::GameResult::Loss,
            _ => crate::GameResult::Draw,
        }
    }
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_sfen_value_size() {
        assert_eq!(std::mem::size_of::<PackedSfenValue>(), 40);
    }

    #[test]
    fn test_packed_sfen_size() {
        assert_eq!(std::mem::size_of::<PackedSfen>(), 32);
    }

    #[test]
    fn test_bitstream_read_bits() {
        let data = [0b10101010, 0b11001100];
        let mut stream = BitStream::new(&data);

        // 0b10101010 の bit0-3 = 0,1,0,1 → result に bit 順に格納 → 0b1010 = 10
        assert_eq!(stream.read_bits(4), 0b1010);

        // 0b10101010 の bit4-7 = 0,1,0,1 → 0b1010 = 10
        assert_eq!(stream.read_bits(4), 0b1010);

        // 0b11001100 の bit0-3 = 0,0,1,1 → 0b1100 = 12
        assert_eq!(stream.read_bits(4), 0b1100);

        // 0b11001100 の bit4-7 = 0,0,1,1 → 0b1100 = 12
        assert_eq!(stream.read_bits(4), 0b1100);
    }

    #[test]
    fn test_bitstream_cursor() {
        let data = [0u8; 32];
        let mut stream = BitStream::new(&data);

        assert_eq!(stream.cursor(), 0);
        stream.read_bit();
        assert_eq!(stream.cursor(), 1);
        stream.read_bits(7);
        assert_eq!(stream.cursor(), 8);
        stream.read_bits(7);
        assert_eq!(stream.cursor(), 15);
    }

    #[test]
    fn test_packed_sfen_value_accessors() {
        let mut psv = PackedSfenValue::default();
        // score = 0x1234 at bytes 32-33
        psv.data[32] = 0x34;
        psv.data[33] = 0x12;
        // game_ply = 0x0056 at bytes 36-37
        psv.data[36] = 0x56;
        psv.data[37] = 0x00;
        // game_result = -1 at byte 38
        psv.data[38] = 0xFF; // -1 as i8

        assert_eq!(psv.score(), 0x1234);
        assert_eq!(psv.game_ply(), 0x0056);
        assert_eq!(psv.game_result(), -1);
    }

    #[test]
    fn test_packed_sfen_value_set_score_roundtrip() {
        let mut psv = PackedSfenValue::default();
        psv.data[36] = 0x56; // game_ply — set_score が隣接 field を壊さないことの確認用
        for s in [0i16, 1, -1, 4144, -4144, i16::MAX, i16::MIN] {
            psv.set_score(s);
            assert_eq!(psv.score(), s);
        }
        assert_eq!(psv.game_ply(), 0x0056);
    }

    #[test]
    fn test_huffman_decode_empty() {
        // 空マス: 0
        let data = [0b00000000u8; 32];
        let mut stream = BitStream::new(&data);

        let piece = decode_board_piece(&mut stream);
        assert_eq!(piece, Piece::NONE);
        assert_eq!(stream.cursor(), 1);
    }

    #[test]
    fn test_huffman_decode_pawn() {
        // 先手歩: 01 (PAWN) + 0 (不成) + 0 (先手) = 0001
        // LSB first: bit0=1, bit1=0, bit2=0, bit3=0 → 0b0001
        let data = [0b00000001u8, 0u8, 0u8, 0u8];
        let mut stream = BitStream::new(&data);

        let piece = decode_board_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Pawn);
        assert_eq!(piece.color, Color::Black);
    }

    #[test]
    fn test_huffman_decode_promoted_pawn() {
        // 後手と金: 01 (PAWN) + 1 (成り) + 1 (後手) = 1101
        // LSB first: bit0=1, bit1=0, bit2=1, bit3=1 → 0b1101
        let data = [0b00001101u8, 0u8, 0u8, 0u8];
        let mut stream = BitStream::new(&data);

        let piece = decode_board_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::ProPawn);
        assert_eq!(piece.color, Color::White);
    }

    #[test]
    fn test_huffman_decode_gold() {
        // 先手金: 01111 (GOLD) + 0 (先手) = 001111
        // LSB first: 1,1,1,1,0,0 → 0b001111
        let data = [0b00001111u8, 0u8, 0u8, 0u8];
        let mut stream = BitStream::new(&data);

        let piece = decode_board_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Gold);
        assert_eq!(piece.color, Color::Black);
    }

    #[test]
    fn test_bitstream_oob_protection() {
        // 2バイト = 16ビットのデータ
        let data = [0xFF, 0xFF];
        let mut stream = BitStream::new(&data);

        // 16ビット読み出し可能
        for _ in 0..16 {
            assert!(stream.can_read());
            stream.read_bit();
        }

        // 17ビット目以降は読み出し不可（false を返す）
        assert!(!stream.can_read());
        assert!(!stream.read_bit()); // OOB だが panic しない
        assert!(!stream.read_bit());
    }

    #[test]
    fn test_decode_hand_piece_pawn() {
        // 手駒の歩: 盤上歩の Huffman 符号 "01" から bit0 を除いた "0" + 成りbit(0) + 先後bit(0)
        // = 00 (2ビット)
        // LSB first: bit0=0, bit1=0 → 0b00
        let data = [0b00000000u8; 32];
        let mut stream = BitStream::new(&data);

        let (piece, is_piecebox) = decode_hand_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Pawn);
        assert_eq!(piece.color, Color::Black);
        assert!(!is_piecebox);
    }

    #[test]
    fn test_decode_hand_piece_gold() {
        // 手駒の金: 盤上金の Huffman 符号 "01111" から bit0 を除いた "0111" + 先後bit(0)
        // 0111 (code) = bit0=1, bit1=1, bit2=1, bit3=0
        // 先後bit = bit4 = 0 (Black)
        // バイト: 0b000_0_0111 = 0b00000111 = 7
        let data = [0b00000111u8; 32];
        let mut stream = BitStream::new(&data);

        let (piece, is_piecebox) = decode_hand_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Gold);
        assert_eq!(piece.color, Color::Black);
        assert!(!is_piecebox);
    }

    #[test]
    fn test_decode_hand_piece_piecebox_pawn() {
        // 駒箱の歩: "0" (歩の手駒符号) + 成りbit(1=駒箱) + 先後bit(0)
        // 0 (code) = bit0 = 0
        // 成りbit = bit1 = 1 (駒箱)
        // 先後bit = bit2 = 0 (Black)
        // バイト: 0b00000_0_1_0 = 0b00000010 = 2
        let data = [0b00000010u8; 32];
        let mut stream = BitStream::new(&data);

        let (piece, is_piecebox) = decode_hand_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Pawn);
        assert_eq!(piece.color, Color::Black);
        assert!(is_piecebox); // 駒箱フラグが立っている
    }

    #[test]
    fn test_decode_hand_piece_piecebox_rook() {
        // 駒箱の飛: 盤上飛の Huffman "111111" から bit0 除去 → "11111" + 成りbit(1=駒箱) + 先後bit(0)
        // 11111 (code) = bit0-4 = 1,1,1,1,1
        // 成りbit = bit5 = 1 (駒箱)
        // 先後bit = bit6 = 0 (Black)
        // バイト: 0b0_0_1_11111 = 0b00111111 = 63
        let data = [0b00111111u8; 32];
        let mut stream = BitStream::new(&data);

        let (piece, is_piecebox) = decode_hand_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Rook);
        assert_eq!(piece.color, Color::Black);
        assert!(is_piecebox);
    }

    #[test]
    fn test_for_each_board_piece_matches_pieces_loop() {
        // for_each_board_piece は board を 1-pass で走査し
        // `(piece_type, color, ascending square)` 順に供給する。本テストは
        // 同じ順序を `pieces(color, pt)` の 26 通り loop で組み立てた reference と
        // 完全一致 (順序込み) であることを確認する。
        // 局面: 黒 9 歩 + 白 9 歩 + 黒銀 1 + 白龍 1 + 両玉 (King は emit 対象外)。
        let mut board = ShogiBoard {
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);
        for file in 0..9 {
            board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
            board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }
        board.board[Square::new(0, 7).index()] = Piece::new(Color::Black, PieceType::Silver);
        board.board[Square::new(8, 1).index()] = Piece::new(Color::White, PieceType::Dragon);

        let mut via_loop: Vec<(Piece, Square)> = Vec::new();
        for &pt in &BOARD_PIECE_TYPES {
            for color in [Color::Black, Color::White] {
                for sq in board.pieces(color, pt) {
                    via_loop.push((Piece::new(color, pt), sq));
                }
            }
        }

        let mut via_helper: Vec<(Piece, Square)> = Vec::new();
        board.for_each_board_piece(|p, sq| via_helper.push((p, sq)));

        assert_eq!(via_loop, via_helper);
        // King は emit 対象外であることを確認。
        assert!(
            via_helper
                .iter()
                .all(|(p, _)| p.piece_type != PieceType::King)
        );
        // 玉以外の駒数は 9 + 9 + 1 + 1 = 20。
        assert_eq!(via_helper.len(), 20);
    }

    #[test]
    fn test_for_each_board_piece_propawn_above_nifu_limit() {
        // と金 (ProPawn) は二歩規制対象外で同一色片側 18 枚まで合法的に到達可能。
        // helper が同一 bucket に 18 個積む経路を踏み、emit 順は `pieces(color, pt)`
        // を 26 通り loop で呼ぶ reference と一致することを確認する。
        let mut board = ShogiBoard {
            black_king_sq: Square::new(0, 8),
            white_king_sq: Square::new(8, 0),
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);
        // 9 筋 × 2 段 = 18 マスに黒 ProPawn を配置 (二歩規制は ProPawn には適用
        // されないので合法局面として表現可)。
        for file in 0..9 {
            board.board[Square::new(file, 5).index()] =
                Piece::new(Color::Black, PieceType::ProPawn);
            board.board[Square::new(file, 6).index()] =
                Piece::new(Color::Black, PieceType::ProPawn);
        }

        let mut via_loop: Vec<(Piece, Square)> = Vec::new();
        for &pt in &BOARD_PIECE_TYPES {
            for color in [Color::Black, Color::White] {
                for sq in board.pieces(color, pt) {
                    via_loop.push((Piece::new(color, pt), sq));
                }
            }
        }
        let mut via_helper: Vec<(Piece, Square)> = Vec::new();
        board.for_each_board_piece(|p, sq| via_helper.push((p, sq)));

        assert_eq!(via_helper.len(), 18);
        assert_eq!(via_loop, via_helper);
    }

    #[test]
    fn test_for_each_board_piece_empty_board() {
        // 全 None の board は何も emit しない。
        let board = ShogiBoard::default();
        let mut emitted = 0;
        board.for_each_board_piece(|_, _| emitted += 1);
        assert_eq!(emitted, 0);
    }

    #[test]
    fn test_hand_decode_cursor_256_boundary() {
        // 256ビット境界のテスト
        // 手駒ループは cursor < 256 でループするので、
        // 256ビットちょうどで終了することを確認

        let data = [0u8; 32];
        // 全て空マス符号 "0" で埋める（1ビット×256=256ビット）
        // これにより cursor が 256 に到達して終了

        let mut stream = BitStream::new(&data);

        // 256ビット読み出し
        for _ in 0..256 {
            stream.read_bit();
        }
        assert_eq!(stream.cursor(), 256);
        assert!(!stream.can_read()); // これ以上読めない

        // OOB アクセスしても panic しない
        assert!(!stream.read_bit());
    }

    // ---------------------------------------------------------------------
    // Fast-Huffman LUT の parity tests
    //
    // `legacy_decode_*` は HUFFMAN_TABLE を 1-bit ループ + 全 entry 線形 scan で
    // 復号する素直な reference 実装 (LUT-free)。LUT 経由の `decode_*` と全
    // 6-bit (board) / 5-bit (hand) peek × promote × color の組合せで
    // bit-identical (decoded piece + 消費 cursor 一致) であることを exhaustive に
    // 確認するための double-check 経路。
    // ---------------------------------------------------------------------

    fn legacy_decode_board_piece(stream: &mut BitStream) -> Piece {
        let mut code = 0u32;
        let mut bits = 0u8;
        loop {
            code |= (stream.read_bit() as u32) << bits;
            bits += 1;
            for (idx, &(pattern, len)) in HUFFMAN_TABLE.iter().enumerate() {
                if bits == len && code == pattern {
                    if idx == 0 {
                        return Piece::NONE;
                    }
                    let base_pt = huffman_index_to_piece_type(idx as u8);
                    let promoted = if idx as u8 != HUFFMAN_GOLD {
                        stream.read_bit()
                    } else {
                        false
                    };
                    let color = if stream.read_bit() {
                        Color::White
                    } else {
                        Color::Black
                    };
                    let pt = if promoted { base_pt.promote() } else { base_pt };
                    return Piece::new(color, pt);
                }
            }
            if bits > 6 {
                return Piece::NONE;
            }
        }
    }

    fn legacy_decode_hand_piece(stream: &mut BitStream) -> (Piece, bool) {
        let mut code = 0u32;
        let mut bits = 0u8;
        loop {
            code |= (stream.read_bit() as u32) << bits;
            bits += 1;
            for (idx, &(pattern, len)) in HUFFMAN_TABLE.iter().enumerate() {
                if idx == 0 {
                    continue;
                }
                let hand_pattern = pattern >> 1;
                let hand_len = len - 1;
                if bits == hand_len && code == hand_pattern {
                    let base_pt = huffman_index_to_piece_type(idx as u8);
                    let is_piecebox = if idx as u8 != HUFFMAN_GOLD {
                        stream.read_bit()
                    } else {
                        false
                    };
                    let color = if stream.read_bit() {
                        Color::White
                    } else {
                        Color::Black
                    };
                    return (Piece::new(color, base_pt), is_piecebox);
                }
            }
            if bits > 5 {
                return (Piece::NONE, false);
            }
        }
    }

    /// 5 byte (40 bit) ある任意の入力で `decode_board_piece` の結果と消費 bit 数
    /// が legacy 1-bit ループ実装と完全一致することを 6-bit peek + 2 flag bit の
    /// 全 256 組合せで確認する。
    #[test]
    fn fast_huffman_board_matches_legacy_all_inputs() {
        // 8 bit (= 6 peek + 2 flag) の全入力を 5 byte buffer の先頭に置き、
        // 残り byte は 0 padding。
        for v in 0u16..256 {
            let bytes = [v as u8, 0u8, 0u8, 0u8, 0u8];

            let mut s_legacy = BitStream::new(&bytes);
            let p_legacy = legacy_decode_board_piece(&mut s_legacy);

            let mut s_fast = BitStream::new(&bytes);
            let p_fast = decode_board_piece(&mut s_fast);

            assert_eq!(p_legacy, p_fast, "piece mismatch at v={v:08b}");
            assert_eq!(
                s_legacy.cursor(),
                s_fast.cursor(),
                "cursor mismatch at v={v:08b}: legacy={} fast={}",
                s_legacy.cursor(),
                s_fast.cursor(),
            );
        }
    }

    #[test]
    fn fast_huffman_hand_matches_legacy_all_inputs() {
        // 7 bit (= 5 peek + 2 flag) の全入力を 5 byte buffer 先頭に置く。
        for v in 0u16..128 {
            let bytes = [v as u8, 0u8, 0u8, 0u8, 0u8];

            let mut s_legacy = BitStream::new(&bytes);
            let (p_legacy, box_legacy) = legacy_decode_hand_piece(&mut s_legacy);

            let mut s_fast = BitStream::new(&bytes);
            let (p_fast, box_fast) = decode_hand_piece(&mut s_fast);

            assert_eq!(p_legacy, p_fast, "piece mismatch at v={v:08b}");
            assert_eq!(box_legacy, box_fast, "piecebox flag mismatch at v={v:08b}");
            assert_eq!(
                s_legacy.cursor(),
                s_fast.cursor(),
                "cursor mismatch at v={v:08b}"
            );
        }
    }

    /// `tests/data/sample.psv` (100 records) を新旧両方の経路で decode し、
    /// `ShogiBoard` の盤面 / 持駒 / 手番 / 玉位置が record 毎 byte-identical
    /// であることを確認する。`from_packed_sfen` の上層 (king pos 読み出し / loop
    /// 終端 / `is_piecebox` の hand 振り分け) も含めた end-to-end parity test。
    #[test]
    fn fast_huffman_full_decode_matches_legacy_on_sample_psv() {
        use std::path::PathBuf;
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/sample.psv");
        // fixture が消えた状態で silent pass にすると parity claim が崩れるので
        // hard fail させる (`tests/data/sample.psv` は repo 同梱、`psv_smoke` も
        // 同じ file を required に扱う)。
        let bytes =
            std::fs::read(&path).expect("crates/shogi-format/tests/data/sample.psv が読めない");
        assert_eq!(bytes.len() % 40, 0);
        // SAFETY:
        // - `PackedSfenValue` は `#[repr(C)]` / size=40 / align=1 で `[u8; 40]` 単体の
        //   POD (`size_of` test で検証済)、任意のバイト列が valid 表現。
        // - `bytes` は `Vec<u8>` の所有データで align=1、`bytes.len() % 40 == 0` を
        //   直前で確認済なので末端 partial record は無い。
        // - 返す slice の lifetime は外側 `bytes` の lifetime 内に閉じる。
        let records: &[PackedSfenValue] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const PackedSfenValue, bytes.len() / 40)
        };
        for (i, psv) in records.iter().enumerate() {
            let fast = ShogiBoard::from_packed_sfen(psv);
            let slow = legacy_from_packed_sfen(psv);
            assert_eq!(fast.board, slow.board, "record {i}: board mismatch");
            // Hand に PartialEq が無いので公開フィールド counts で比較する。
            assert_eq!(
                fast.black_hand.counts, slow.black_hand.counts,
                "record {i}: black_hand"
            );
            assert_eq!(
                fast.white_hand.counts, slow.white_hand.counts,
                "record {i}: white_hand"
            );
            assert_eq!(
                fast.side_to_move, slow.side_to_move,
                "record {i}: side_to_move"
            );
            assert_eq!(
                fast.black_king_sq, slow.black_king_sq,
                "record {i}: black_king_sq"
            );
            assert_eq!(
                fast.white_king_sq, slow.white_king_sq,
                "record {i}: white_king_sq"
            );
        }
    }

    fn legacy_from_packed_sfen(psv: &PackedSfenValue) -> ShogiBoard {
        let mut board = ShogiBoard {
            score: psv.score(),
            result: psv.game_result(),
            ply: psv.game_ply(),
            ..Default::default()
        };
        let mut stream = BitStream::new(&psv.sfen().data);

        board.side_to_move = if stream.read_bit() {
            Color::White
        } else {
            Color::Black
        };
        let black_king_idx = stream.read_bits(7) as u8;
        let white_king_idx = stream.read_bits(7) as u8;
        board.black_king_sq = Square(black_king_idx);
        board.white_king_sq = Square(white_king_idx);
        if black_king_idx < 81 {
            board.board[black_king_idx as usize] = Piece::new(Color::Black, PieceType::King);
        }
        if white_king_idx < 81 {
            board.board[white_king_idx as usize] = Piece::new(Color::White, PieceType::King);
        }
        for sq_idx in 0..81u8 {
            if sq_idx == black_king_idx || sq_idx == white_king_idx {
                continue;
            }
            board.board[sq_idx as usize] = legacy_decode_board_piece(&mut stream);
        }
        while stream.cursor() < 256 {
            let (piece, is_piecebox) = legacy_decode_hand_piece(&mut stream);
            if is_piecebox {
                continue;
            }
            let pt = piece.piece_type;
            match piece.color {
                Color::Black => board.black_hand.add(pt, 1),
                Color::White => board.white_hand.add(pt, 1),
            }
        }
        board
    }

    #[test]
    fn peek_bits_zero_pads_past_end() {
        // バッファ末尾を越えた peek_bits は 0 padding で返ることを確認。
        // (decode loop が cursor 終端を制御する前提で OOB panic させない。)
        let data = [0xFFu8, 0xFFu8];
        let stream = BitStream::new(&data);
        // cursor=0 で 16 bit 全部読めることを peek で確認 (mask 通すと 0xFFFF)。
        assert_eq!(stream.peek_bits(16), 0xFFFF);

        // バッファ末尾ちょうど (cursor=16, byte_pos == data.len()) で peek
        let mut s2 = BitStream::new(&data);
        s2.advance(16);
        assert_eq!(s2.peek_bits(6), 0); // 全て 0 padding

        let mut s3 = BitStream::new(&data);
        s3.advance(15);
        // cursor 15: data[1] の bit 7 (=1) + 5 bit padding (=0) → 0b000001
        assert_eq!(s3.peek_bits(6), 0b000001);

        // bit_cursor が bit_limit を大きく越えた (advance 連発の後 etc.、
        // byte_pos > data.len()) ケースでも slice 起点 clamp により panic
        // しないことを確認。
        let mut s4 = BitStream::new(&data);
        s4.advance(64); // bit_cursor=64 → byte_pos=8、data.len()=2
        assert_eq!(s4.peek_bits(6), 0);
        assert_eq!(s4.peek_bits(25), 0);
    }
}
