//! 将棋 NNUE 用の基本型と PackedSfenValue デコーダ。
//!
//! 型定義 / バイト並びの出典は bullet-shogi のオリジナル実装
//! (`ATTRIBUTION.md` 参照)。

pub mod bona_piece;
pub mod game_result;
pub mod packed_sfen;
pub mod types;

pub use bona_piece::BonaPiece;
pub use game_result::GameResult;
pub use packed_sfen::{BitStream, PackedSfen, PackedSfenValue, ShogiBoard};
pub use types::{Color, Hand, Piece, PieceType, Square};
