//! experiments/001-cuda-oxide-kpabs の dummy entry point。
//!
//! 本 binary は scaffold (Issue #8) — PSV を 1 batch 読み込み、先頭数 record の
//! 主要フィールド (score / game_ply / game_result) を print するだけ。
//! GPU は触らない (Issue #9 以降で kernel + host loop を増築していく)。
//!
//! ## 使い方
//!
//! ```bash
//! # 引数なし: shogi-format crate の test fixture (sample.psv, 100 records) を読む
//! cargo run -p exp-001-cuda-oxide-kpabs
//!
//! # 引数あり: 任意の PSV file path を渡す
//! cargo run -p exp-001-cuda-oxide-kpabs -- /path/to/data.psv
//! ```

use std::env;
use std::fs;
use std::mem::size_of;
use std::path::PathBuf;
use std::process::ExitCode;

use shogi_format::PackedSfenValue;

const PSV_SIZE: usize = size_of::<PackedSfenValue>();

fn main() -> ExitCode {
    let path = match env::args_os().nth(1) {
        Some(p) => PathBuf::from(p),
        None => default_sample_path(),
    };

    // 表示用に `..` を畳んで見やすくする (失敗したら raw のまま)
    let display_path = path.canonicalize().unwrap_or_else(|_| path.clone());
    println!("reading PSV from: {}", display_path.display());
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: failed to read {}: {e}", path.display());
            return ExitCode::from(2);
        }
    };

    if bytes.len() % PSV_SIZE != 0 {
        eprintln!(
            "error: file size {} is not a multiple of PSV record size {PSV_SIZE}",
            bytes.len()
        );
        return ExitCode::from(2);
    }

    let count = bytes.len() / PSV_SIZE;
    println!("file size: {} bytes / {count} records", bytes.len());

    // SAFETY: `PackedSfenValue` は `#[repr(C)] struct { data: [u8; 40] }` で
    // alignment は 1。`Vec<u8>` の as_ptr() は alignment 1 を満たし、上の
    // size 検査 (`bytes.len() % PSV_SIZE == 0`) で N records 分のメモリが
    // 連続して読み出し可能。同パターンは shogi-format/tests/psv_smoke.rs
    // の `read_one_batch_of_psv_records` で invariant を verifying 済み。
    let records: &[PackedSfenValue] =
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const PackedSfenValue, count) };

    let take = count.min(5);
    for (i, psv) in records.iter().take(take).enumerate() {
        println!(
            "[{i}] score={:>6} ply={:>4} game_result={:>2} ({:?})",
            psv.score(),
            psv.game_ply(),
            psv.game_result(),
            psv.result()
        );
    }
    if count > take {
        println!("... ({} more records)", count - take);
    }

    ExitCode::SUCCESS
}

/// 引数省略時に読む、shogi-format crate の test fixture。
///
/// experiments/001-cuda-oxide-kpabs/ → ../../crates/shogi-format/tests/data/sample.psv
fn default_sample_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/shogi-format/tests/data/sample.psv")
}
