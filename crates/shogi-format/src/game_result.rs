//! 対局結果。discriminant は `PackedSfenValue` の game_result encoding と
//! 一致 (Loss=0, Draw=1, Win=2)。

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GameResult {
    Loss = 0,
    Draw = 1,
    Win = 2,
}
