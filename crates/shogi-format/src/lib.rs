//! 将棋 NNUE 用の基本型と PackedSfenValue デコーダ。

pub mod bona_piece;
pub mod game_result;
pub mod huffman_coded_pos;
pub mod packed_sfen;
pub mod types;

pub use bona_piece::BonaPiece;
pub use game_result::GameResult;
pub use huffman_coded_pos::{HCPE_RECORD_BYTES, HuffmanCodedPosAndEval};
pub use packed_sfen::{BitStream, PackedSfen, PackedSfenValue, ShogiBoard};
pub use types::{Color, Hand, Piece, PieceType, Square};
