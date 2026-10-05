//! 教師データの decode 済み局面に対する構造検証。

use std::io;
use std::path::Path;

use shogi_format::ShogiBoard;
use shogi_format::types::{Color, HAND_PIECE_TYPES, PieceType, Square};

/// 玉の盤上配置と king square fields、盤上と持ち駒の総数を確認する。
/// 駒落ちは許可するが、教師データには両陣営の玉が必要。完全な合法性検証はしない。
/// PSV decode は unchecked で、壊れた record も盤面へ変換され得るため検証する。
pub(crate) fn validate_board(board: &ShogiBoard) -> Result<(), String> {
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

/// 元ファイルの record 番号がわかる読み出しでは、その0始まりの番号を診断に含める。
/// 元の番号を保持しない学習 worker の入力は path のみを付け、番号を推測しない。
pub(crate) fn validate_psv_board(
    board: &ShogiBoard,
    path: &Path,
    record_index: Option<u64>,
) -> io::Result<()> {
    validate_board(board).map_err(|reason| {
        let location = match record_index {
            Some(index) => format!("record {index} in {}", path.display()),
            None => format!("PSV input {}", path.display()),
        };
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{location} decodes to a corrupt position ({reason})"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataloader::Batch;
    use shogi_features::FeatureSet;
    use shogi_format::{PackedSfenValue, Piece};

    fn sample_board() -> ShogiBoard {
        let bytes = include_bytes!("../../shogi-format/tests/data/sample.psv");
        let mut psv = PackedSfenValue::default();
        psv.as_bytes_mut().copy_from_slice(&bytes[..40]);
        psv.decode()
    }

    #[test]
    fn teacher_validation_accepts_handicap_with_both_kings() {
        let mut board = sample_board();
        assert!(validate_board(&board).is_ok());
        let piece = board
            .board
            .iter_mut()
            .find(|piece| !matches!(piece.piece_type, PieceType::None | PieceType::King))
            .expect("sample has non-king pieces");
        *piece = Piece::NONE;
        assert!(validate_board(&board).is_ok());
        let mut batch = Batch::with_capacity(1, FeatureSet::HalfKaHmMerged.spec());
        assert!(batch.push_decoded(&board).unwrap());
        assert!(batch.nnz[0] > 0);
    }

    #[test]
    fn teacher_contract_does_not_change_generic_one_king_features() {
        let mut board = sample_board();
        board.board[board.black_king_sq.index()] = Piece::NONE;
        board.black_king_sq = Square::NONE;
        assert!(validate_board(&board).is_err());
        let mut batch = Batch::with_capacity(1, FeatureSet::HalfKaHmMerged.spec());
        assert!(batch.push_decoded(&board).unwrap());
        assert_eq!(batch.nnz[0], 0);
    }
}
