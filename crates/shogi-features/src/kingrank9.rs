//! LayerStack の KingRank9 bucket 割当。

use shogi_format::{Color, ShogiBoard};

/// KingRank9 の固定 bucket 数。
pub const KINGRANK9_NUM_BUCKETS: usize = 9;

/// YaneuraOu LayerStacks の KingRank9 バケット (`0..=8`)。
///
/// 双方の玉段を手番側視点に正規化し、3 段刻みで 3x3 に分割する。
pub fn kingrank9_bucket_board(board: &ShogiBoard) -> u8 {
    debug_assert!(board.black_king_sq.is_valid());
    debug_assert!(board.white_king_sq.is_valid());

    let (friendly_king, enemy_king) = match board.side_to_move {
        Color::Black => (board.black_king_sq, board.white_king_sq.inverse()),
        Color::White => (board.white_king_sq.inverse(), board.black_king_sq),
    };

    let friendly_group = friendly_king.rank() / 3;
    let enemy_group = enemy_king.rank() / 3;
    friendly_group * 3 + enemy_group
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_format::types::Square;

    fn board_from_sfen(sfen: &str) -> ShogiBoard {
        let mut fields = sfen.split_whitespace();
        let placement = fields.next().expect("SFEN board field");
        let side_to_move = match fields.next().expect("SFEN side-to-move field") {
            "b" => Color::Black,
            "w" => Color::White,
            side => panic!("unexpected SFEN side to move: {side}"),
        };
        let mut board = ShogiBoard {
            side_to_move,
            ..Default::default()
        };
        for (rank, row) in placement.split('/').enumerate() {
            let mut file_from_left = 0usize;
            for ch in row.chars() {
                if let Some(empty) = ch.to_digit(10) {
                    file_from_left += empty as usize;
                    continue;
                }
                if ch == '+' {
                    continue;
                }
                let file = 8 - file_from_left;
                match ch {
                    'K' => board.black_king_sq = Square::new(file as u8, rank as u8),
                    'k' => board.white_king_sq = Square::new(file as u8, rank as u8),
                    _ => {}
                }
                file_from_left += 1;
            }
            assert_eq!(file_from_left, 9, "SFEN rank must contain nine squares");
        }
        assert!(
            board.black_king_sq.is_valid(),
            "SFEN must contain black king"
        );
        assert!(
            board.white_king_sq.is_valid(),
            "SFEN must contain white king"
        );
        board
    }

    #[test]
    fn oracle_fixtures_match_yaneuraou_kingrank9() {
        let fixtures = [
            // 先手玉 5i (rank=8) → kF=6、後手玉 5a を反転 (rank=8) → kE=2。
            (
                "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
                8,
            ),
            // 後手玉 5b を反転 (rank=7) → kF=6、先手玉 5h (rank=7) → kE=2。
            (
                "lnsg1gsnl/1r2k2b1/ppppppppp/9/9/9/PPPPPPPPP/1B2K2R1/LNSG1GSNL w - 1",
                8,
            ),
            // 先手玉 5c (rank=2) → kF=0、後手玉 5g を反転 (rank=2) → kE=0。
            ("9/9/4K4/9/9/9/4k4/9/9 b - 1", 0),
            // 先手玉 5d (rank=3) → kF=3、後手玉 5e を反転 (rank=4) → kE=1。
            ("9/9/9/4K4/4k4/9/9/9/9 b - 1", 4),
            // 後手玉 5a を反転 (rank=8) → kF=6、先手玉 5i (rank=8) → kE=2。
            (
                "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
                8,
            ),
        ];

        for (sfen, expected) in fixtures {
            assert_eq!(
                kingrank9_bucket_board(&board_from_sfen(sfen)),
                expected,
                "{sfen}"
            );
        }
    }
}
