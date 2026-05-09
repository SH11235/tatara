//! 対局結果。
//!
//! bullet-shogi `crate::value::loader::GameResult` の最小サブセット。
//! discriminant は bullet と互換 (Loss=0, Draw=1, Win=2)。

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GameResult {
    Loss = 0,
    Draw = 1,
    Win = 2,
}
