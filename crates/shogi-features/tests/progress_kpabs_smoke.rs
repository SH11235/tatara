//! ShogiProgressKPAbs の smoke test。
//!
//! shogi-format crate の `tests/data/sample.psv` (100 records) を共有して、
//! 各 record で `for_each_active_index` / `collect_active_indices` /
//! `progress` / `bucket` が妥当な値を返すことを確認する。
//!
//! 重み (`progress.bin`) は意図的にロードしないため、
//! `weights()` は zero 配列、`progress()` は常に `sigmoid(0) = 0.5`、
//! `bucket()` は `floor(0.5 * 8) = 4` になる。

use shogi_features::{SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, ShogiProgressKPAbs};
use shogi_format::PackedSfenValue;
use std::fs;
use std::mem::size_of;
use std::path::PathBuf;

const SAMPLE_RECORD_COUNT: usize = 100;
const PSV_SIZE: usize = 40;

fn sample_records() -> Vec<PackedSfenValue> {
    // shogi-format crate と fixture を共有。
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../shogi-format/tests/data/sample.psv");
    let bytes = fs::read(&path).expect("sample.psv が読めない (../shogi-format/tests/data/)");
    assert_eq!(bytes.len(), SAMPLE_RECORD_COUNT * PSV_SIZE);
    assert_eq!(size_of::<PackedSfenValue>(), PSV_SIZE);

    let records: &[PackedSfenValue] = unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const PackedSfenValue,
            bytes.len() / PSV_SIZE,
        )
    };
    records.to_vec()
}

#[test]
fn for_each_active_index_yields_in_range() {
    let records = sample_records();
    for (i, psv) in records.iter().enumerate() {
        let mut count = 0usize;
        ShogiProgressKPAbs::for_each_active_index(psv, |idx| {
            assert!(
                idx < SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS,
                "record {i}: index {idx} >= NUM_WEIGHTS={SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS}"
            );
            count += 1;
        });

        // 平手初期から終盤まで、各局面で 王 + 駒 → indices は二桁以上は出るはず
        // (両玉の特徴が両 perspective で 0 になるとしても、駒は ≥ 1 セットで複数 index)。
        assert!(
            count >= 2,
            "record {i}: active indices が {count} 個しか無い (王 + 駒で最低 2 を期待)"
        );
    }
}

#[test]
fn collect_active_indices_matches_for_each() {
    let records = sample_records();
    let mut buf = Vec::new();
    for (i, psv) in records.iter().enumerate() {
        let mut via_for_each = Vec::new();
        ShogiProgressKPAbs::for_each_active_index(psv, |idx| via_for_each.push(idx));

        ShogiProgressKPAbs::collect_active_indices(psv, &mut buf);

        assert_eq!(
            buf, via_for_each,
            "record {i}: collect_active_indices と for_each_active_index の出力が不一致"
        );
    }
}

#[test]
fn progress_and_bucket_with_zero_weights() {
    // 重み未ロード時の global state 前提:
    //   weights = [0.0; N]  →  Σ w_i x_i = 0  →  sigmoid(0) = 0.5
    //   bucket = floor(0.5 * 8.0) = 4
    //
    // CAVEAT: 本テストは `SHOGI_PROGRESS_KP_ABS_WEIGHTS` (OnceLock) が
    // この test binary プロセス内で **一度も .set() されていない** ことに
    // 依存する。各 `tests/*.rs` ファイルは独立 binary に compile されるので
    // 他 test file からの汚染は起きないが、本 file 内に
    // `ShogiProgressKPAbs::load_from_bin` を呼ぶ test を追加するときは
    // 順序依存で本テストが壊れる点に注意。本格的な loaded-weights テストは
    // 別 binary (別 tests/*.rs) に分離すること。
    let kpabs = ShogiProgressKPAbs;
    let records = sample_records();

    for (i, psv) in records.iter().enumerate() {
        let p = kpabs.progress(psv);
        assert!(
            (0.0..=1.0).contains(&p),
            "record {i}: progress {p} が [0,1] 外"
        );
        // 浮動小数の sigmoid(0) は厳密に 0.5 になるはずだが念のため近傍 chk
        assert!(
            (p - 0.5).abs() < 1e-6,
            "record {i}: weights=0 で progress {p} が 0.5 から離れている"
        );

        let b = kpabs.bucket(psv);
        assert_eq!(b, 4, "record {i}: weights=0 で bucket={b} (期待: 4)");
        assert!(b < 8, "bucket {b} が 0..=7 範囲外");
    }
}
