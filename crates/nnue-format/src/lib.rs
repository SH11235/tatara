//! `nnue-format` crate — NNUE binary serialization (header + halfka_psqt weights)。
//!
//! Stage 3 (EPIC #17) の output 形式 = rshogi 互換 NNUE binary を扱う。
//! 設計方針は本 crate を **GPU 非依存・pure CPU library** に保ち、CI workflow
//! でも `--exclude` せず test を通すこと。trainer (`bins/nnue_train`) は
//! Stage 3-8 (#65) で本 crate の save API を呼んで weight を吐き出す。
//!
//! ## 提供予定 module
//!
//! - `header` (Stage 3-2, #58): NNUE binary の先頭 metadata
//!   (net_id / FV_SCALE / QA / QB) の (de)serialise
//! - `halfka_psqt` (Stage 3-3, #59): HalfKA_hm 1536-16-32 weight の
//!   quantised save / float load。bullet-shogi `crates/bullet_lib/src/value/
//!   save.rs` (commit `f275eb9`) を vendor、rshogi loader と互換確保
//!
//! scaffold (Stage 3-0, #56) 段階では module を持たず、後続 issue で追加。
