//! `nnue-format` crate — NNUE binary serialization (header + halfka_psqt weights)。
//!
//! Stage 3 (EPIC #17) の output 形式 = rshogi 互換 NNUE binary を扱う。
//! 設計方針は本 crate を **GPU 非依存・pure CPU library** に保ち、CI workflow
//! でも `--exclude` せず test を通すこと。trainer (`bins/nnue_train`) は
//! Stage 3-8 (#65) で本 crate の save API を呼んで weight を吐き出す。
//!
//! ## 提供 module
//!
//! - `header` (Stage 3-2, #58): NNUE binary の先頭 22 bytes 固定長 metadata
//!   (`NnueHeader`: net_id / fv_scale / qa / qb) の (de)serialise
//! - `halfka_psqt` (Stage 3-3, #59): HalfKA_hm + PSQT NNUE binary (FT + L1 +
//!   PSQT) の `save_quantised` / `load`、bullet 上流 `crates/trainer/src/
//!   model/save.rs::QuantTarget` の量子化ロジックを移植

pub mod halfka_psqt;
pub mod header;

pub use halfka_psqt::{FT_OUT_DIM, HalfKAPsqtNet, L1_OUT_DIM, NUM_FEATURES, QuantTarget};
pub use header::{DEFAULT_FV_SCALE, DEFAULT_QA, DEFAULT_QB, HEADER_BYTES, NET_ID_LEN, NnueHeader};
