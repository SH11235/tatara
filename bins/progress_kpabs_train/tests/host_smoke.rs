//! host helper の単体テスト (GPU 非依存)。
//!
//! host loop の純粋ロジック (Batch builder / PSV reader / GameIterator /
//! progress.bin I/O) のみを検証する。
//!
//! ## 使う test fixture
//!
//! - `crates/shogi-format/tests/data/sample.psv` (100 records、bullet-shogi
//!   smoke_progress 由来の先頭 4000 bytes を vendor したもの)

use std::path::PathBuf;

use progress_kpabs_train::host::MAX_INDS_PER_POS;
use progress_kpabs_train::host::batch::Batch;
use progress_kpabs_train::host::games::{GameIterator, PackCursor};
use progress_kpabs_train::host::progress_bin::{read_progress_bin, write_progress_bin};
use shogi_features::SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS;

fn sample_psv_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/shogi-format/tests/data/sample.psv")
}

#[test]
fn pack_cursor_reads_all_records() {
    let path = sample_psv_path();
    let mut cursor = PackCursor::open(&path).expect("open sample.psv");
    let initial = cursor.remaining();
    assert_eq!(initial, 100, "sample.psv は 100 records");

    let mut count = 0_u64;
    while let Some(_psv) = cursor.next_psv().expect("read") {
        count += 1;
    }
    assert_eq!(count, 100);
    assert_eq!(cursor.remaining(), 0);
}

#[test]
fn game_iterator_splits_by_ply_decrease() {
    // sample.psv は 100 records が 1 game ぶんの sfen 進行 (game_ply 単調増加)
    // のはずなので、GameIterator は 1 ゲーム = 100 records を返す想定。
    let path = sample_psv_path();
    let cursor = PackCursor::open(&path).expect("open");
    let mut gi = GameIterator::new(cursor);

    let first = gi.next_game().expect("next").expect("Some");
    assert!(
        !first.is_empty(),
        "最初のゲームは空ではない (got {} records)",
        first.len()
    );
    // game_ply が単調増加であること (sample.psv は 1 ゲーム想定)
    let mut prev = 0_u16;
    for psv in &first {
        assert!(
            psv.game_ply() >= prev,
            "game_ply 単調増加: prev={prev} cur={}",
            psv.game_ply()
        );
        prev = psv.game_ply();
    }

    // 2 ゲーム目があるなら最初の record の ply は 1 ゲーム目末尾より小さいはず
    if let Some(second) = gi.next_game().expect("next") {
        assert!(
            second[0].game_ply() < prev,
            "次ゲーム先頭は ply 減少: cur_first={} prev_last={}",
            second[0].game_ply(),
            prev
        );
    }
}

#[test]
fn game_iterator_splits_on_equal_ply_boundary() {
    // bullet-shogi 上流は `cur_ply <= prev` を境界とするため、ply が等しい場合も
    // 新ゲーム先頭になる (例: 1 局を 50 手目で打ち切って次局も 50 手目から始まる
    // ような edge ケース)。本テストは sample.psv の先頭 1 record と重複した
    // PSV を合成して、ply == prev の境界が正しく分割されることを確認する。
    let path = sample_psv_path();
    let mut cursor = PackCursor::open(&path).expect("open");
    let r0 = cursor.next_psv().expect("read").expect("Some");
    let r1 = cursor.next_psv().expect("read").expect("Some");

    // 一時 PSV ファイルを作って [r0, r0_dup_with_same_ply] を流し込む。
    // r0_dup は r0 そのまま (game_ply は同じ) — つまり境界の == 条件をテスト。
    let dir = tempdir();
    let synthetic = dir.join("synthetic_equal_ply.psv");
    let mut bytes = Vec::with_capacity(80);
    bytes.extend_from_slice(r0.as_bytes());
    bytes.extend_from_slice(r1.as_bytes()); // ply は r0 より大きいはず → 同ゲーム
    bytes.extend_from_slice(r0.as_bytes()); // ply == r0 だが r1 直後 → 新ゲーム想定
    std::fs::write(&synthetic, &bytes).expect("write");

    let cursor = PackCursor::open(&synthetic).expect("open synthetic");
    let mut gi = GameIterator::new(cursor);
    let first = gi.next_game().expect("next").expect("first");
    let second = gi.next_game().expect("next").expect("second");
    assert_eq!(first.len(), 2, "前ゲームは r0+r1 の 2 records");
    assert_eq!(
        second.len(),
        1,
        "後ゲームは ply == r0_ply で先頭の 1 record"
    );
    assert_eq!(
        second[0].game_ply(),
        r0.game_ply(),
        "境界 ply は r0 と等しい"
    );
}

#[test]
fn batch_push_game_builds_flat_arrays() {
    // 2 ゲーム × 3 position の batch を組んで、flat array の長さと finalize 後の
    // per_pos_norm を確認する。
    let path = sample_psv_path();
    let cursor = PackCursor::open(&path).expect("open");
    let mut gi = GameIterator::new(cursor);
    let game = gi.next_game().expect("next").expect("first game");

    let mut batch = Batch::new();
    let mut scratch = Vec::with_capacity(96);
    // 同じゲームを 2 回 push して 2 games 分にする (簡易合成 batch)
    let take = 3.min(game.len());
    let prefix = game[..take].to_vec();
    batch.push_game(&prefix, &mut scratch);
    batch.push_game(&prefix, &mut scratch);
    batch.finalize();

    assert_eq!(batch.n_games, 2);
    assert_eq!(batch.n_positions, 2 * take);
    assert_eq!(batch.indices.len(), 2 * take * MAX_INDS_PER_POS);
    assert_eq!(batch.targets.len(), 2 * take);
    assert_eq!(batch.per_pos_norm.len(), 2 * take);
    // finalize 後の per_pos_norm = 1 / (game_len * n_games) = 1 / (take * 2)
    let expected_norm = 1.0_f32 / (take as f32 * 2.0_f32);
    for &n in &batch.per_pos_norm {
        assert!(
            (n - expected_norm).abs() < 1e-7,
            "per_pos_norm = {n} expected {expected_norm}"
        );
    }
}

#[test]
fn batch_targets_are_linear_per_game() {
    // game_len = 5 で push_game した target は [0, 0.25, 0.5, 0.75, 1.0]
    let path = sample_psv_path();
    let cursor = PackCursor::open(&path).expect("open");
    let mut gi = GameIterator::new(cursor);
    let game = gi.next_game().expect("next").expect("first");
    let trimmed = game[..5.min(game.len())].to_vec();
    let mut batch = Batch::new();
    let mut scratch = Vec::new();
    batch.push_game(&trimmed, &mut scratch);

    assert_eq!(batch.targets.len(), trimmed.len());
    if trimmed.len() == 5 {
        let expected = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
        for (i, (got, exp)) in batch.targets.iter().zip(&expected).enumerate() {
            assert!(
                (got - exp).abs() < 1e-7,
                "target[{i}] = {got} expected {exp}"
            );
        }
    }
}

#[test]
fn progress_bin_roundtrip() {
    let dir = tempdir();
    let path = dir.join("progress.bin");

    // 全 weight が 0.0 のベクトルで roundtrip
    let weights = vec![0.0_f32; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS];
    write_progress_bin(&path, &weights).expect("write");
    let read = read_progress_bin(&path).expect("read");
    assert_eq!(read.len(), weights.len());
    for &w in &read {
        assert_eq!(w, 0.0_f32);
    }

    // 一部に値を入れて roundtrip
    let mut weights2 = vec![0.0_f32; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS];
    weights2[0] = 1.5;
    weights2[100] = -2.25;
    weights2[SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS - 1] = 3.125;
    write_progress_bin(&path, &weights2).expect("write");
    let read2 = read_progress_bin(&path).expect("read");
    assert_eq!(read2[0], 1.5);
    assert_eq!(read2[100], -2.25);
    assert_eq!(read2[SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS - 1], 3.125);
}

#[test]
fn progress_bin_size_is_exactly_1003104_bytes() {
    // progress.bin: f64 LE × N_WEIGHTS = 1_003_104 bytes 固定。
    let dir = tempdir();
    let path = dir.join("progress.bin");
    let weights = vec![0.0_f32; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS];
    write_progress_bin(&path, &weights).expect("write");
    let size = std::fs::metadata(&path).expect("stat").len();
    let expected = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS as u64 * std::mem::size_of::<f64>() as u64;
    assert_eq!(size, expected);
    assert_eq!(size, 1_003_104);
}

#[test]
fn progress_bin_rejects_wrong_size() {
    let dir = tempdir();
    let path = dir.join("bogus.bin");
    std::fs::write(&path, [0_u8; 16]).expect("write bogus");
    let err = read_progress_bin(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

/// 一時ディレクトリ helper (tempfile crate の dep を避けるため最小実装)。
fn tempdir() -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("rshogi-nnue-stage1-9-{pid}-{nanos}"));
    std::fs::create_dir_all(&dir).expect("mkdir temp");
    dir
}
