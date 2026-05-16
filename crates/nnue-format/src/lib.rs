//! `nnue-format` crate — NNUE binary serialization (header + weights)。
//!
//! 量子化 NNUE binary (LayerStack 形式 + HalfKA_hm+PSQT 形式) の
//! (de)serialize を担う **GPU 非依存・pure CPU library**。trainer
//! (`bins/nnue_train`) が weight を量子化して書き出す際に呼ぶ。
//!
//! ## 提供 module
//!
//! - `header`: NNUE binary の先頭 22 bytes 固定長 metadata
//!   (`NnueHeader`: net_id / fv_scale / qa / qb) の (de)serialise
//! - `halfka_psqt`: HalfKA_hm + PSQT NNUE binary (FT + L1 + PSQT) の
//!   `save_quantised` / `load`
//! - `layerstack_weights`: LayerStack quantised binary
//!   (HalfKA_hm + 9-bucket LayerStack) の save / load

pub mod halfka_psqt;
pub mod header;
pub mod layerstack_weights;

pub use halfka_psqt::{FT_OUT_DIM, HalfKAPsqtNet, L1_OUT_DIM, NUM_FEATURES, QuantTarget};
pub use header::{DEFAULT_FV_SCALE, DEFAULT_QA, DEFAULT_QB, HEADER_BYTES, NET_ID_LEN, NnueHeader};
pub use layerstack_weights::LayerStackWeights;
