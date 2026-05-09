//! 将棋 NNUE 用の基本型と PackedSfenValue デコーダ。
//!
//! bullet-shogi (commit `f275eb9`) の `crates/bullet_lib/src/shogi/` から
//! vendor。取り込み元・差分は本リポジトリの `ATTRIBUTION.md` を参照。

pub mod bona_piece;
pub mod game_result;
pub mod packed_sfen;
pub mod types;

pub use bona_piece::BonaPiece;
pub use game_result::GameResult;
pub use packed_sfen::{BitStream, PackedSfen, PackedSfenValue, ShogiBoard};
pub use types::{Color, Hand, Piece, PieceType, Square};
