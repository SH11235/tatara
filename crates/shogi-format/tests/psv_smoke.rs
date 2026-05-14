//! PSV reader smoke test。
//!
//! `tests/data/sample.psv` は外部 PSV データ 1 file の先頭 100 records
//! (40 byte × 100 = 4000 bytes) を切り出した固定 fixture。
//! 1 batch 分の PSV を read & decode できることを確認する。

use shogi_format::{Color, GameResult, PackedSfenValue};
use std::fs;
use std::mem::size_of;
use std::path::PathBuf;

const SAMPLE_RECORD_COUNT: usize = 100;
const PSV_SIZE: usize = 40;

fn sample_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/sample.psv")
}

#[test]
fn sample_has_expected_size() {
    let path = sample_path();
    let bytes = fs::read(&path).expect("tests/data/sample.psv が読めない");
    assert_eq!(
        bytes.len(),
        SAMPLE_RECORD_COUNT * PSV_SIZE,
        "sample.psv は 40 bytes × {} records = {} bytes であるべき",
        SAMPLE_RECORD_COUNT,
        SAMPLE_RECORD_COUNT * PSV_SIZE,
    );
    // PSV size invariant が drift していないことの二重確認
    assert_eq!(size_of::<PackedSfenValue>(), PSV_SIZE);
}

#[test]
fn read_one_batch_of_psv_records() {
    let path = sample_path();
    let bytes = fs::read(&path).expect("tests/data/sample.psv が読めない");

    // PackedSfenValue は #[repr(C)] / 40 byte 固定で transmute 可
    let records: &[PackedSfenValue] = unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const PackedSfenValue,
            bytes.len() / PSV_SIZE,
        )
    };
    assert_eq!(records.len(), SAMPLE_RECORD_COUNT);

    // 全 record で公開アクセサが panic せず、値が妥当範囲か確認
    for (i, psv) in records.iter().enumerate() {
        let ply = psv.game_ply();
        assert!(ply <= 1024, "record {i}: game_ply {ply} が想定上限を超えた");

        let score = psv.score();
        // mate score 込みの実用上限として ±32600 を採用 (sample は ±32000 内に収まる)。
        assert!(
            (-32_600..=32_600).contains(&score),
            "record {i}: score {score} が想定範囲外 (mate 含む ±32600)"
        );

        // raw game_result は i8 (-1..=+1)
        let raw = psv.game_result();
        assert!((-1..=1).contains(&raw), "record {i}: raw game_result {raw}");

        // result() は Win/Loss/Draw 必ずいずれか (panic しない)
        let _ = psv.result();
    }

    // 先頭 record の board decode が走り、最低限の合法性を持つ
    let first = &records[0];
    let board = first.decode();

    // 王はいずれの手番にも 1 枚ずつ盤上に存在する。
    // Square::NONE (=81) や 81 以上の不正値が紛れていないことを確認する
    // (decode が読み出す king index は 7 bit = 0..=127 で raw に Square に
    // 入るため、`is_valid()` (=index < 81) で本物の盤上 square かを検査する)。
    for color in [Color::Black, Color::White] {
        let sq = board.king_square(color);
        assert!(
            sq.is_valid(),
            "record 0 ({color:?}): king square index={} が盤外",
            sq.index()
        );
    }
}

#[test]
fn game_result_enum_discriminants_match_bullet() {
    // 上流 (bullet-shogi) と同じ discriminant: Loss=0, Draw=1, Win=2。
    assert_eq!(GameResult::Loss as u8, 0);
    assert_eq!(GameResult::Draw as u8, 1);
    assert_eq!(GameResult::Win as u8, 2);
}
