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

use cuda_device::{DisjointSlice, kernel, thread};
use shogi_format::PackedSfenValue;

const PSV_SIZE: usize = size_of::<PackedSfenValue>();

/// Forward kernel (KP-abs sigmoid prediction per position).
///
/// **本 binary は kernel を直接 launch しない**。`#[kernel]` を main.rs に
/// inline 定義しているのは、cuda-oxide の rustc-codegen-cuda backend が
/// **bin entry から到達可能な kernel** を PTX 化する設計だから (lib.rs 内
/// kernel は `cargo oxide build <crate>` では PTX に出ない、本リポでは
/// 経験的に未生成を確認)。GPU launch path は Stage 1-9 (#13) で host loop
/// を組むときに ここから呼び出す。
///
/// アルゴリズムと bullet-shogi 上流 (`KERNELS_SRC::k_forward`) との差分は
/// reference CPU 実装 (`src/kernels/forward.rs::forward_cpu`、lib path で
/// `exp_001_cuda_oxide_kpabs::kernels::forward::forward_cpu`) の docstring
/// および `ATTRIBUTION.md` の Stage 1-5 entry を参照。
#[kernel]
pub fn forward(
    indices: &[i32],
    weights: &[f32],
    mut preds: DisjointSlice<f32>,
    n_pos: u32,
    max_inds: u32,
) {
    let pos = thread::index_1d();
    if pos.get() >= n_pos as usize {
        return;
    }
    let mut z = 0.0f32;
    let base = pos.get() * (max_inds as usize);
    let mut j: u32 = 0;
    while j < max_inds {
        let idx = indices[base + (j as usize)];
        if idx >= 0 {
            z += weights[idx as usize];
        }
        j += 1;
    }
    if let Some(p) = preds.get_mut(pos) {
        *p = 1.0f32 / (1.0f32 + (-z).exp());
    }
}

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
