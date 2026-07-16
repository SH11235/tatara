//! Apery / dlshogi HuffmanCodedPosAndEval decoder.

use std::io;

use super::packed_sfen::{BitStream, ShogiBoard};
use super::types::{Color, Piece, PieceType, Square};

/// HCPE record size in bytes.
pub const HCPE_RECORD_BYTES: usize = 38;

#[derive(Clone, Copy)]
struct Code {
    pattern: u8,
    bits: u8,
    piece: Piece,
}

const fn code(pattern: u8, bits: u8, color: Color, piece_type: PieceType) -> Code {
    Code {
        pattern,
        bits,
        piece: Piece::new(color, piece_type),
    }
}

const BOARD_CODES: [Code; 27] = [
    code(0b0, 1, Color::Black, PieceType::None),
    code(0b1, 4, Color::Black, PieceType::Pawn),
    code(0b11, 6, Color::Black, PieceType::Lance),
    code(0b111, 6, Color::Black, PieceType::Knight),
    code(0b1011, 6, Color::Black, PieceType::Silver),
    code(0b1_1111, 8, Color::Black, PieceType::Bishop),
    code(0b11_1111, 8, Color::Black, PieceType::Rook),
    code(0b1111, 6, Color::Black, PieceType::Gold),
    code(0b1001, 4, Color::Black, PieceType::ProPawn),
    code(0b10_0011, 6, Color::Black, PieceType::ProLance),
    code(0b10_0111, 6, Color::Black, PieceType::ProKnight),
    code(0b10_1011, 6, Color::Black, PieceType::ProSilver),
    code(0b1001_1111, 8, Color::Black, PieceType::Horse),
    code(0b1011_1111, 8, Color::Black, PieceType::Dragon),
    code(0b101, 4, Color::White, PieceType::Pawn),
    code(0b1_0011, 6, Color::White, PieceType::Lance),
    code(0b1_0111, 6, Color::White, PieceType::Knight),
    code(0b1_1011, 6, Color::White, PieceType::Silver),
    code(0b0101_1111, 8, Color::White, PieceType::Bishop),
    code(0b0111_1111, 8, Color::White, PieceType::Rook),
    code(0b10_1111, 6, Color::White, PieceType::Gold),
    code(0b1101, 4, Color::White, PieceType::ProPawn),
    code(0b11_0011, 6, Color::White, PieceType::ProLance),
    code(0b11_0111, 6, Color::White, PieceType::ProKnight),
    code(0b11_1011, 6, Color::White, PieceType::ProSilver),
    code(0b1101_1111, 8, Color::White, PieceType::Horse),
    code(0b1111_1111, 8, Color::White, PieceType::Dragon),
];

const HAND_CODES: [Code; 14] = [
    code(0b0, 3, Color::Black, PieceType::Pawn),
    code(0b1, 5, Color::Black, PieceType::Lance),
    code(0b11, 5, Color::Black, PieceType::Knight),
    code(0b101, 5, Color::Black, PieceType::Silver),
    code(0b111, 5, Color::Black, PieceType::Gold),
    code(0b1_1111, 7, Color::Black, PieceType::Bishop),
    code(0b11_1111, 7, Color::Black, PieceType::Rook),
    code(0b100, 3, Color::White, PieceType::Pawn),
    code(0b1_0001, 5, Color::White, PieceType::Lance),
    code(0b1_0011, 5, Color::White, PieceType::Knight),
    code(0b1_0101, 5, Color::White, PieceType::Silver),
    code(0b1_0111, 5, Color::White, PieceType::Gold),
    code(0b101_1111, 7, Color::White, PieceType::Bishop),
    code(0b111_1111, 7, Color::White, PieceType::Rook),
];

fn decode_code(stream: &mut BitStream<'_>, codes: &[Code]) -> io::Result<Piece> {
    let peek = stream.peek_bits(8) as u8;
    for entry in codes {
        let mask = (1u16 << entry.bits) - 1;
        if u16::from(peek) & mask == u16::from(entry.pattern) {
            stream.advance(entry.bits as usize);
            return Ok(entry.piece);
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "invalid HuffmanCodedPos prefix",
    ))
}

/// One 38-byte HuffmanCodedPosAndEval record.
#[derive(Clone, Copy)]
pub struct HuffmanCodedPosAndEval {
    data: [u8; HCPE_RECORD_BYTES],
}

impl Default for HuffmanCodedPosAndEval {
    fn default() -> Self {
        Self {
            data: [0; HCPE_RECORD_BYTES],
        }
    }
}

impl HuffmanCodedPosAndEval {
    pub fn as_bytes_mut(&mut self) -> &mut [u8; HCPE_RECORD_BYTES] {
        &mut self.data
    }

    pub fn score(&self) -> i16 {
        i16::from_le_bytes([self.data[32], self.data[33]])
    }

    pub fn game_result(&self) -> u8 {
        self.data[36]
    }

    /// Decode the record into the representation shared by feature extraction.
    pub fn decode(&self) -> io::Result<ShogiBoard> {
        let mut stream = BitStream::new(&self.data[..32]);
        let side_to_move = if stream.read_bit() {
            Color::White
        } else {
            Color::Black
        };
        let black_king_idx = stream.read_bits(7) as u8;
        let white_king_idx = stream.read_bits(7) as u8;
        if black_king_idx >= 81 || white_king_idx >= 81 || black_king_idx == white_king_idx {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid king squares in HuffmanCodedPos",
            ));
        }

        let absolute_result = self.game_result();
        let result = match (absolute_result, side_to_move) {
            (0, _) => 0,
            (1, Color::Black) | (2, Color::White) => 1,
            (1, Color::White) | (2, Color::Black) => -1,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid HCPE game result {absolute_result}"),
                ));
            }
        };

        let mut board = ShogiBoard {
            side_to_move,
            black_king_sq: Square(black_king_idx),
            white_king_sq: Square(white_king_idx),
            score: self.score(),
            result,
            ..Default::default()
        };
        board.board[black_king_idx as usize] = Piece::new(Color::Black, PieceType::King);
        board.board[white_king_idx as usize] = Piece::new(Color::White, PieceType::King);

        for sq_idx in 0..81u8 {
            if sq_idx != black_king_idx && sq_idx != white_king_idx {
                board.board[sq_idx as usize] = decode_code(&mut stream, &BOARD_CODES)?;
            }
        }
        while stream.cursor() < 256 {
            let piece = decode_code(&mut stream, &HAND_CODES)?;
            match piece.color {
                Color::Black => board.black_hand.add(piece.piece_type, 1),
                Color::White => board.white_hand.add(piece.piece_type, 1),
            }
        }
        if stream.cursor() != 256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "HuffmanCodedPos does not end on its 256-bit boundary",
            ));
        }

        // 駒枚数 inventory の検証。将棋は全 40 駒が盤上か手駒に必ず存在し (駒落ちは対象外)、
        // 成りを戻した総数は固定在庫に一致する。同じ符号長のコード取り違え (例: 香 6bit ↔
        // 桂 6bit) は後続オフセットと 256-bit 境界を保つため上流検査をすり抜けるが、在庫が
        // 崩れるためここで捕捉する。counts は PieceType discriminant で index。
        let mut inv = [0u16; 15];
        for piece in &board.board {
            if piece.is_some() {
                inv[piece.piece_type.unpromote() as usize] += 1;
            }
        }
        // Hand.counts の並びは pawn, lance, knight, silver, gold, bishop, rook。
        const HAND_SLOT: [PieceType; 7] = [
            PieceType::Pawn,
            PieceType::Lance,
            PieceType::Knight,
            PieceType::Silver,
            PieceType::Gold,
            PieceType::Bishop,
            PieceType::Rook,
        ];
        for hand in [&board.black_hand, &board.white_hand] {
            for (slot, &count) in HAND_SLOT.iter().zip(hand.counts.iter()) {
                inv[*slot as usize] += u16::from(count);
            }
        }
        const EXPECTED: [(PieceType, u16); 8] = [
            (PieceType::Pawn, 18),
            (PieceType::Lance, 4),
            (PieceType::Knight, 4),
            (PieceType::Silver, 4),
            (PieceType::Gold, 4),
            (PieceType::Bishop, 2),
            (PieceType::Rook, 2),
            (PieceType::King, 2),
        ];
        for (pt, want) in EXPECTED {
            if inv[pt as usize] != want {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "HuffmanCodedPos piece inventory invalid: {:?} count {} (expected {want})",
                        pt, inv[pt as usize]
                    ),
                ));
            }
        }

        Ok(board)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PackedSfenValue;

    #[test]
    fn record_size_is_38_bytes() {
        assert_eq!(std::mem::size_of::<HuffmanCodedPosAndEval>(), 38);
    }

    // 中盤局面: 後手番 + 持ち駒 7 種 12 枚 + 成駒 2 枚。ply-1 fixture が踏まない
    // 全 hand code (14) と成駒 board code、後手番デコードを網羅する回帰ガード。
    // hcpe と psv は同一局面の別エンコードで、両デコードが一致すること = HuffmanCodedPos
    // の hand/成駒テーブルが正しいこと。局面の由来はコメント末尾参照。
    #[test]
    fn decodes_midgame_hand_and_promoted_same_as_psv() {
        let hcpe_bytes = [
            0x8d, 0xb8, 0x09, 0x15, 0x06, 0x00, 0x00, 0x80, 0x85, 0xf0, 0x6e, 0x4a, 0xfc, 0x62,
            0x2b, 0xe1, 0x45, 0x89, 0xe3, 0x13, 0xfe, 0x5e, 0x61, 0xa2, 0x04, 0x03, 0x00, 0x60,
            0x8c, 0x0f, 0x67, 0xbd, 0xe9, 0x07, 0x2b, 0x1a, 0x02, 0x00,
        ];
        let psv_bytes = [
            0x8d, 0xb8, 0x11, 0x19, 0x06, 0x00, 0x00, 0x80, 0x83, 0xf0, 0x9e, 0x52, 0xfc, 0xe1,
            0x4c, 0xe1, 0x45, 0x8a, 0xe5, 0x0b, 0x7e, 0x5f, 0x61, 0x24, 0x05, 0xc3, 0xa3, 0x14,
            0x00, 0xc0, 0x9d, 0x95, 0xe9, 0x07, 0x2b, 0x1a, 0x01, 0x00, 0x01, 0x00,
        ];
        let mut hcpe = HuffmanCodedPosAndEval::default();
        hcpe.as_bytes_mut().copy_from_slice(&hcpe_bytes);
        let mut psv = PackedSfenValue::default();
        psv.as_bytes_mut().copy_from_slice(&psv_bytes);

        let actual = hcpe.decode().expect("decode HCPE midgame fixture");
        let expected = psv.decode();
        assert_eq!(actual.board, expected.board);
        assert_eq!(actual.black_hand.counts, expected.black_hand.counts);
        assert_eq!(actual.white_hand.counts, expected.white_hand.counts);
        assert_eq!(actual.side_to_move, expected.side_to_move);
        assert_eq!(actual.black_king_sq, expected.black_king_sq);
        assert_eq!(actual.white_king_sq, expected.white_king_sq);
        assert_eq!(actual.score, expected.score);
        assert_eq!(actual.result, expected.result);

        // この fixture が意図した経路 (後手番/持ち駒/成駒) を実際に踏むことを固定。
        assert_eq!(actual.side_to_move, crate::Color::White);
        let hand_types = actual.black_hand.counts.iter().filter(|&&c| c > 0).count()
            + actual.white_hand.counts.iter().filter(|&&c| c > 0).count();
        assert!(hand_types >= 3, "fixture must exercise multiple hand codes");
        assert!(
            actual
                .board
                .iter()
                .filter(|p| p.is_some() && p.piece_type.is_promoted())
                .count()
                >= 2,
            "fixture must exercise promoted board codes"
        );
    }

    #[test]
    fn decodes_same_position_and_labels_as_corresponding_psv() {
        let hcpe_bytes = [
            0x8c, 0xae, 0x09, 0x15, 0x06, 0x00, 0x00, 0x80, 0x85, 0xf0, 0x6e, 0x4a, 0xbc, 0x6c,
            0x25, 0x80, 0x12, 0xc7, 0xa7, 0x3f, 0x1c, 0xc2, 0x44, 0x09, 0x06, 0x00, 0xc0, 0xf9,
            0x70, 0xce, 0x7a, 0xff, 0x86, 0xf8, 0xb8, 0x2b, 0x02, 0x00,
        ];
        let psv_bytes = [
            0x8c, 0xae, 0x11, 0x19, 0x06, 0x00, 0x00, 0x80, 0x83, 0xf0, 0x9e, 0x52, 0xbc, 0x9c,
            0x29, 0x80, 0x14, 0xcb, 0x97, 0x5f, 0x2c, 0xc2, 0x48, 0x0a, 0x86, 0xc7, 0x01, 0x00,
            0x7c, 0xef, 0xac, 0x95, 0x86, 0xf8, 0xb8, 0x43, 0x01, 0x00, 0xff, 0x00,
        ];
        let mut hcpe = HuffmanCodedPosAndEval::default();
        hcpe.as_bytes_mut().copy_from_slice(&hcpe_bytes);
        let mut psv = PackedSfenValue::default();
        psv.as_bytes_mut().copy_from_slice(&psv_bytes);

        let actual = hcpe.decode().expect("decode HCPE fixture");
        let expected = psv.decode();
        assert_eq!(actual.board, expected.board);
        assert_eq!(actual.black_hand.counts, expected.black_hand.counts);
        assert_eq!(actual.white_hand.counts, expected.white_hand.counts);
        assert_eq!(actual.side_to_move, expected.side_to_move);
        assert_eq!(actual.black_king_sq, expected.black_king_sq);
        assert_eq!(actual.white_king_sq, expected.white_king_sq);
        assert_eq!(actual.score, expected.score);
        assert_eq!(actual.result, expected.result);
    }

    fn legal_inventory(b: &ShogiBoard) -> bool {
        let mut inv = [0u16; 15];
        for p in &b.board {
            if p.is_some() {
                inv[p.piece_type.unpromote() as usize] += 1;
            }
        }
        const HAND_SLOT: [PieceType; 7] = [
            PieceType::Pawn,
            PieceType::Lance,
            PieceType::Knight,
            PieceType::Silver,
            PieceType::Gold,
            PieceType::Bishop,
            PieceType::Rook,
        ];
        for hand in [&b.black_hand, &b.white_hand] {
            for (slot, &c) in HAND_SLOT.iter().zip(hand.counts.iter()) {
                inv[*slot as usize] += u16::from(c);
            }
        }
        inv[PieceType::Pawn as usize] == 18
            && inv[PieceType::Lance as usize] == 4
            && inv[PieceType::Knight as usize] == 4
            && inv[PieceType::Silver as usize] == 4
            && inv[PieceType::Gold as usize] == 4
            && inv[PieceType::Bishop as usize] == 2
            && inv[PieceType::Rook as usize] == 2
            && inv[PieceType::King as usize] == 2
    }

    // 同符号長コードの取り違えは 256-bit 境界を保つため境界検査を抜ける。全 256 bit に
    // 1-bit 反転を走らせ、(1) デコード成功した盤は必ず合法在庫であること、(2) 少なくとも
    // 1 つは reject されること (= inventory / 境界検査が到達可能) を固定する。
    #[test]
    fn corrupted_records_never_yield_illegal_inventory() {
        let base: [u8; 38] = [
            0x8d, 0xb8, 0x09, 0x15, 0x06, 0x00, 0x00, 0x80, 0x85, 0xf0, 0x6e, 0x4a, 0xfc, 0x62,
            0x2b, 0xe1, 0x45, 0x89, 0xe3, 0x13, 0xfe, 0x5e, 0x61, 0xa2, 0x04, 0x03, 0x00, 0x60,
            0x8c, 0x0f, 0x67, 0xbd, 0xe9, 0x07, 0x2b, 0x1a, 0x02, 0x00,
        ];
        let mut any_rejected = false;
        for bit in 0..256usize {
            let mut bytes = base;
            bytes[bit / 8] ^= 1 << (bit % 8);
            let mut h = HuffmanCodedPosAndEval::default();
            h.as_bytes_mut().copy_from_slice(&bytes);
            match h.decode() {
                Ok(b) => assert!(
                    legal_inventory(&b),
                    "decode returned board with illegal inventory (flipped bit {bit})"
                ),
                Err(_) => any_rejected = true,
            }
        }
        assert!(any_rejected, "no corruption was rejected");
    }
}
