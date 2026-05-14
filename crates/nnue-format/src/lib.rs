//! `nnue-format` crate — NNUE binary serialization (header + weights)。
//!
//! rshogi 互換 NNUE binary を扱う **GPU 非依存・pure CPU library**。trainer
//! (`bins/nnue_train`) が weight を量子化して書き出す際に呼ぶ。
//!
//! ## 提供 module
//!
//! - `header`: NNUE binary の先頭 22 bytes 固定長 metadata
//!   (`NnueHeader`: net_id / fv_scale / qa / qb) の (de)serialise
//! - `halfka_psqt`: HalfKA_hm + PSQT NNUE binary (FT + L1 + PSQT) の
//!   `save_quantised` / `load`
//! - `v102_layerstack`: bullet v102 互換 LayerStack binary
//!   (HalfKA_hm + 9-bucket LayerStack) の save / load

pub mod halfka_psqt;
pub mod header;
pub mod v102_layerstack;

pub use halfka_psqt::{FT_OUT_DIM, HalfKAPsqtNet, L1_OUT_DIM, NUM_FEATURES, QuantTarget};
pub use header::{DEFAULT_FV_SCALE, DEFAULT_QA, DEFAULT_QB, HEADER_BYTES, NET_ID_LEN, NnueHeader};
pub use v102_layerstack::V102Weights;
