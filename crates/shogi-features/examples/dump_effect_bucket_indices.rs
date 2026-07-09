use std::env;
use std::fs::File;
use std::io::{self, Write};

use shogi_features::{EffectBucketConfig, collect_effect_bucket_features_board};
use shogi_format::ShogiBoard;
use shogi_format::types::{Color, Hand, Piece, PieceType, Square};

const SFENS: [&str; 5] = [
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
    "l4S2l/4g1gs1/5p1p1/pr2N1pkp/4Gn3/PP3PPPP/2GPP4/1K7/L3r+s2L w BS2N5Pb 1",
    "6n1l/2+S1k4/2lp4p/1np1B2b1/3PP4/1N1S3rP/1P2+pPP+p1/1p1G5/3KG2r1 b GSN2L4Pgs2p 1",
    "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
    "lnsgkgsnl/1r5b1/ppppppppp/9/4P4/9/PPPP1PPPP/1B5R1/LNSGKGSNL b - 1",
];

const CONFIGS: [(&str, EffectBucketConfig); 4] = [
    (
        "effect_bucket_2x2_kingfixed",
        EffectBucketConfig::KINGFIXED_2X2,
    ),
    (
        "effect_bucket_2x2_kingbucketed",
        EffectBucketConfig::KINGBUCKETED_2X2,
    ),
    (
        "effect_bucket_3x3_kingfixed",
        EffectBucketConfig::KINGFIXED_3X3,
    ),
    (
        "effect_bucket_3x3_kingbucketed",
        EffectBucketConfig::KINGBUCKETED_3X3,
    ),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut out = output_writer()?;
    for (sfen_no, sfen) in SFENS.iter().enumerate() {
        let board = parse_sfen(sfen)?;
        for perspective in [Color::Black, Color::White] {
            for &(config_name, config) in &CONFIGS {
                let mut indices = collect_effect_bucket_features_board(&board, config, perspective);
                indices.sort_unstable();
                write!(
                    out,
                    "{sfen_no} {} {config_name} :",
                    perspective_label(perspective)
                )?;
                for idx in indices {
                    write!(out, " {idx}")?;
                }
                writeln!(out)?;
            }
        }
    }
    Ok(())
}

fn output_writer() -> io::Result<Box<dyn Write>> {
    match env::args_os().nth(1) {
        Some(path) => File::create(path).map(|f| Box::new(f) as Box<dyn Write>),
        None => Ok(Box::new(io::stdout())),
    }
}

fn perspective_label(color: Color) -> &'static str {
    match color {
        Color::Black => "B",
        Color::White => "W",
    }
}

fn parse_sfen(sfen: &str) -> Result<ShogiBoard, String> {
    let mut parts = sfen.split_whitespace();
    let board_part = parts.next().ok_or("missing board")?;
    let side_part = parts.next().ok_or("missing side-to-move")?;
    let hand_part = parts.next().ok_or("missing hand")?;
    let ply_part = parts.next().ok_or("missing ply")?;

    let mut board = ShogiBoard {
        side_to_move: parse_side(side_part)?,
        ply: ply_part
            .parse::<u16>()
            .map_err(|_| format!("invalid ply: {ply_part}"))?,
        ..Default::default()
    };

    parse_board(board_part, &mut board)?;
    parse_hand(hand_part, &mut board)?;
    Ok(board)
}

fn parse_side(s: &str) -> Result<Color, String> {
    match s {
        "b" => Ok(Color::Black),
        "w" => Ok(Color::White),
        _ => Err(format!("invalid side-to-move: {s}")),
    }
}

fn parse_board(board_part: &str, board: &mut ShogiBoard) -> Result<(), String> {
    let ranks: Vec<&str> = board_part.split('/').collect();
    if ranks.len() != 9 {
        return Err(format!("board must have 9 ranks: {board_part}"));
    }

    for (rank, row) in ranks.iter().enumerate() {
        let mut file_from_left = 0usize;
        let mut promoted = false;
        for ch in row.chars() {
            if ch == '+' {
                if promoted {
                    return Err(format!("duplicate promotion marker in rank: {row}"));
                }
                promoted = true;
                continue;
            }
            if let Some(skip) = ch.to_digit(10) {
                if promoted {
                    return Err(format!(
                        "promotion marker before empty squares in rank: {row}"
                    ));
                }
                file_from_left += skip as usize;
                continue;
            }
            if file_from_left >= 9 {
                return Err(format!("too many files in rank: {row}"));
            }
            let color = if ch.is_ascii_uppercase() {
                Color::Black
            } else {
                Color::White
            };
            let mut pt = piece_type_from_sfen(ch)?;
            if promoted {
                pt = pt.promote();
                promoted = false;
            }
            let file = 8 - file_from_left;
            let sq = Square::new(file as u8, rank as u8);
            let piece = Piece::new(color, pt);
            board.board[sq.index()] = piece;
            if pt == PieceType::King {
                match color {
                    Color::Black => board.black_king_sq = sq,
                    Color::White => board.white_king_sq = sq,
                }
            }
            file_from_left += 1;
        }
        if promoted {
            return Err(format!("dangling promotion marker in rank: {row}"));
        }
        if file_from_left != 9 {
            return Err(format!("rank does not contain 9 files: {row}"));
        }
    }

    Ok(())
}

fn parse_hand(hand_part: &str, board: &mut ShogiBoard) -> Result<(), String> {
    if hand_part == "-" {
        return Ok(());
    }

    let mut count = 0u8;
    for ch in hand_part.chars() {
        if let Some(digit) = ch.to_digit(10) {
            count = count
                .saturating_mul(10)
                .saturating_add(u8::try_from(digit).expect("single digit fits u8"));
            continue;
        }
        let n = if count == 0 { 1 } else { count };
        count = 0;
        let color = if ch.is_ascii_uppercase() {
            Color::Black
        } else {
            Color::White
        };
        let pt = piece_type_from_sfen(ch)?;
        if !pt.can_be_in_hand() {
            return Err(format!("piece cannot be in hand: {ch}"));
        }
        hand_mut(board, color).add(pt, n);
    }
    if count != 0 {
        return Err(format!("dangling hand count in: {hand_part}"));
    }
    Ok(())
}

fn hand_mut(board: &mut ShogiBoard, color: Color) -> &mut Hand {
    match color {
        Color::Black => &mut board.black_hand,
        Color::White => &mut board.white_hand,
    }
}

fn piece_type_from_sfen(ch: char) -> Result<PieceType, String> {
    match ch.to_ascii_uppercase() {
        'P' => Ok(PieceType::Pawn),
        'L' => Ok(PieceType::Lance),
        'N' => Ok(PieceType::Knight),
        'S' => Ok(PieceType::Silver),
        'G' => Ok(PieceType::Gold),
        'B' => Ok(PieceType::Bishop),
        'R' => Ok(PieceType::Rook),
        'K' => Ok(PieceType::King),
        _ => Err(format!("invalid piece char: {ch}")),
    }
}
