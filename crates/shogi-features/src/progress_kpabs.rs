//! KP-absolute progress feature。
//!
//! 仕様:
//! - 特徴次元: `81 * FE_OLD_END = 81 * 1548 = 125_388` (玉位置 × BonaPiece)
//! - 学習形式: logistic regression (`p = sigmoid(Σ w_i * x_i)`)
//! - bucket 割当: caller が指定した `num_buckets ∈ [1, u8::MAX as usize + 1]`
//!   に対し `min(num_buckets - 1, floor(p * num_buckets))`。返り値が `u8` の
//!   ため上限は 256 (`bucket = num_buckets - 1` が `u8::MAX` に収まる)
//! - 重み読込: `progress.bin` (f64 little-endian × 125_388 個、`8 * 125_388 = 1_003_104` bytes 固定)
//! - Q16 bucket 割当 (`*_q16` 系): 係数を `round(w * 65536)` の i32、和を i64、閾値を
//!   `round(ln(i / (N - i)) * 65536)` (`i = 1..N`) とする整数版。推論エンジン rshogi の
//!   `progresskpabsq16` が同じ量子化を使う。整数和は加算順に依存しないので、
//!   推論側が差分更新しても学習時と同じ bucket になる

use std::path::Path;
use std::sync::OnceLock;

use shogi_format::bona_piece::FE_OLD_END;
use shogi_format::types::HAND_PIECE_TYPES;
use shogi_format::{BonaPiece, Color, PackedSfenValue, ShogiBoard};

/// KP-absolute 特徴の次元数: `81 * FE_OLD_END`。
pub const SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS: usize = 81 * FE_OLD_END;

static SHOGI_PROGRESS_KP_ABS_WEIGHTS: OnceLock<LoadedWeights> = OnceLock::new();
static SHOGI_PROGRESS_KP_ABS_ZERO_WEIGHTS: [f32; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS] =
    [0.0; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS];
static SHOGI_PROGRESS_KP_ABS_ZERO_WEIGHTS_Q16: [i32; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS] =
    [0; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS];
static Q16_THRESHOLDS: OnceLock<Box<[Box<[i64]>]>> = OnceLock::new();

/// Q16 の量子化は f32 へ丸める前の f64 から行う (f32 経由だと推論側と値がずれる)。
struct LoadedWeights {
    f32: Box<[f32]>,
    q16: Box<[i32]>,
}

/// Progress-based N-bucket assignment using KP-absolute features.
///
/// 重みはプロセス全体で 1 つ (`OnceLock` で保持) のため、本 struct は `Copy`
/// にできる。bucket 数 `N` は caller が指定する (LayerStack `--num-buckets` 等)。
#[derive(Clone, Copy, Default)]
pub struct ShogiProgressKPAbs;

impl ShogiProgressKPAbs {
    fn weights() -> &'static [f32] {
        SHOGI_PROGRESS_KP_ABS_WEIGHTS
            .get()
            .map_or(&SHOGI_PROGRESS_KP_ABS_ZERO_WEIGHTS, |weights| {
                weights.f32.as_ref()
            })
    }

    fn weights_q16() -> &'static [i32] {
        SHOGI_PROGRESS_KP_ABS_WEIGHTS
            .get()
            .map_or(&SHOGI_PROGRESS_KP_ABS_ZERO_WEIGHTS_Q16, |weights| {
                weights.q16.as_ref()
            })
    }

    /// 指定局面の KP-absolute 有効 index を全列挙し、`f` に渡す。
    ///
    /// `progresskpabs` で使う特徴展開そのもの。
    pub fn for_each_active_index(pos: &PackedSfenValue, f: impl FnMut(usize)) {
        Self::for_each_active_index_board(&pos.decode(), f);
    }

    /// `for_each_active_index` の **decode 済み `ShogiBoard` を直接受ける** 版。
    ///
    /// dataloader が 1 局面につき `PackedSfenValue::decode()` を 1 回だけ呼んで
    /// その `ShogiBoard` を HalfKA_hm 特徴抽出と本 progress 計算の両方で使い回す
    /// ための入口 (`for_each_active_index` は `pos.decode()` 経由で本メソッドを
    /// 呼ぶのと等価)。
    pub fn for_each_active_index_board(board: &ShogiBoard, mut f: impl FnMut(usize)) {
        if !board.black_king_sq.is_valid() || !board.white_king_sq.is_valid() {
            return;
        }

        let sq_bk = board.black_king_sq.index();
        let sq_wk = board.white_king_sq.inverse().index();

        board.for_each_board_piece(|piece, sq| {
            let bp_b = BonaPiece::from_piece_square(piece, sq, Color::Black);
            if bp_b != BonaPiece::ZERO {
                f(sq_bk * FE_OLD_END + bp_b.value() as usize);
            }

            let bp_w = BonaPiece::from_piece_square(piece, sq, Color::White);
            if bp_w != BonaPiece::ZERO {
                f(sq_wk * FE_OLD_END + bp_w.value() as usize);
            }
        });

        for owner in [Color::Black, Color::White] {
            let hand = if owner == Color::Black {
                board.black_hand
            } else {
                board.white_hand
            };
            for &pt in &HAND_PIECE_TYPES {
                let count = hand.count(pt);
                for c in 1..=count {
                    let bp_b = BonaPiece::from_hand_piece(Color::Black, owner, pt, c);
                    if bp_b != BonaPiece::ZERO {
                        f(sq_bk * FE_OLD_END + bp_b.value() as usize);
                    }

                    let bp_w = BonaPiece::from_hand_piece(Color::White, owner, pt, c);
                    if bp_w != BonaPiece::ZERO {
                        f(sq_wk * FE_OLD_END + bp_w.value() as usize);
                    }
                }
            }
        }
    }

    /// `for_each_active_index` の結果を `Vec` に集める。`out` は事前 clear される。
    pub fn collect_active_indices(pos: &PackedSfenValue, out: &mut Vec<usize>) {
        out.clear();
        Self::for_each_active_index(pos, |idx| out.push(idx));
    }

    /// `progress.bin` (f64 LE × `NUM_WEIGHTS`、`1_003_104` bytes 固定) を読み込む。
    ///
    /// プロセスでロード可能な KP-absolute モデルは 1 つだけ (二回目以降は Err)。
    pub fn load_from_bin(path: &Path) -> Result<Self, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("failed to read '{}': {e}", path.display()))?;
        Self::load_from_bytes(&bytes)
    }

    /// [`Self::load_from_bin`] のバイト列版。呼び出し側がロード内容の hash 検証を
    /// 行う場合に、検証したバイト列と同一の内容から重みを構築できる (ここで
    /// ファイルを読み直すと、その間の差し替えで hash とロード内容が乖離する)。
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let expected = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * std::mem::size_of::<f64>();
        if bytes.len() != expected {
            return Err(format!(
                "progress.bin size mismatch: got {} bytes, expected {}",
                bytes.len(),
                expected
            ));
        }

        let raw: Vec<f64> = bytes
            .chunks_exact(std::mem::size_of::<f64>())
            .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("chunk size is checked")))
            .collect();
        // 非有限値は f32 routing では sigmoid が NaN になって bucket 0 に、Q16 では飽和
        // キャストで端の値になり、どちらも誤 routing を黙って続けるので読込時に弾く。
        if let Some(i) = raw.iter().position(|w| !w.is_finite()) {
            return Err(format!(
                "progress.bin coefficient {i} is not finite ({})",
                raw[i]
            ));
        }
        let weights = LoadedWeights {
            f32: raw.iter().map(|&w| w as f32).collect(),
            q16: raw.iter().map(|&w| quantize_q16(w)).collect(),
        };

        SHOGI_PROGRESS_KP_ABS_WEIGHTS.set(weights).map_err(|_| {
            "KP-absolute progress weights are already loaded in this process".to_string()
        })?;

        Ok(Self)
    }

    /// progress 推定 (`0.0..=1.0`)。重み未ロードでは常に 0.5 (sigmoid(0))。
    pub fn progress(&self, pos: &PackedSfenValue) -> f32 {
        self.progress_board(&pos.decode())
    }

    /// `progress` の **decode 済み `ShogiBoard` を直接受ける** 版。
    pub fn progress_board(&self, board: &ShogiBoard) -> f32 {
        let weights = Self::weights();
        let mut sum = 0.0f32;
        Self::for_each_active_index_board(board, |idx| sum += weights[idx]);

        let p = 1.0 / (1.0 + (-sum).exp());
        p.clamp(0.0, 1.0)
    }

    /// N-bucket 割当 (`0..=num_buckets-1`)。`num_buckets ∈ [1, MAX_NUM_BUCKETS]`
    /// を assert する (返り値が `u8` のため上限 256)。`num_buckets = 9` で
    /// `floor(p * 9).clamp(0, 8)` を返し、index 8 まで emit する (LayerStack
    /// 既定)。`num_buckets = N` での split 境界は `i / N` (`i = 0..N`) で等間隔。
    pub fn bucket(&self, pos: &PackedSfenValue, num_buckets: usize) -> u8 {
        self.bucket_board(&pos.decode(), num_buckets)
    }

    /// `bucket` の **decode 済み `ShogiBoard` を直接受ける** 版。
    /// `bucket(&pos, n)` は `bucket_board(&pos.decode(), n)` と等価。
    pub fn bucket_board(&self, board: &ShogiBoard, num_buckets: usize) -> u8 {
        assert!(
            (1..=MAX_NUM_BUCKETS).contains(&num_buckets),
            "num_buckets must be in [1, {MAX_NUM_BUCKETS}] (got {num_buckets}); \
             upper bound is `u8::MAX as usize + 1` so `bucket = num_buckets - 1` \
             fits in u8"
        );
        let p = self.progress_board(board);
        // num_buckets <= 256 を assert 済なので `as i32` キャストは無誤差、
        // clamp の上下境界 (i32::MIN..i32::MAX 内) も安全に評価できる。
        let n_i32 = num_buckets as i32;
        let raw = (p * num_buckets as f32).floor() as i32;
        raw.clamp(0, n_i32 - 1) as u8
    }

    /// Q16 logit 和 (`Σ round(w * 65536)`)。重み未ロードでは常に 0。
    pub fn progress_q16_sum_board(&self, board: &ShogiBoard) -> i64 {
        q16_sum_with(board, Self::weights_q16())
    }

    /// Q16 版の N-bucket 割当。`num_buckets` の範囲は [`Self::bucket_board`] と同じ。
    /// 重み未ロード (和 0) では [`Self::bucket_board`] と同じく `floor(N / 2)` を返す。
    pub fn bucket_q16_board(&self, board: &ShogiBoard, num_buckets: usize) -> u8 {
        q16_sum_to_bucket(self.progress_q16_sum_board(board), num_buckets)
    }
}

fn quantize_q16(weight: f64) -> i32 {
    // `as` は i32 の範囲外を端の値へ飽和させる。
    (weight * 65536.0).round() as i32
}

fn q16_sum_with(board: &ShogiBoard, weights: &[i32]) -> i64 {
    let mut sum = 0i64;
    ShogiProgressKPAbs::for_each_active_index_board(board, |idx| sum += i64::from(weights[idx]));
    sum
}

/// Q16 logit 和を bucket へ変換する。閾値と等しい和は上側の bucket に入る
/// (f32 版の `floor(p * N)` で `p = i / N` ちょうどが bucket `i` になるのと同じ向き)。
pub fn q16_sum_to_bucket(sum: i64, num_buckets: usize) -> u8 {
    assert!(
        (1..=MAX_NUM_BUCKETS).contains(&num_buckets),
        "num_buckets must be in [1, {MAX_NUM_BUCKETS}] (got {num_buckets})"
    );
    let thresholds = &Q16_THRESHOLDS.get_or_init(build_q16_thresholds)[num_buckets];
    thresholds.partition_point(|&t| t <= sum) as u8
}

fn build_q16_thresholds() -> Box<[Box<[i64]>]> {
    (0..=MAX_NUM_BUCKETS)
        .map(|n| {
            (1..n)
                .map(|i| {
                    let p = i as f64 / n as f64;
                    ((p / (1.0 - p)).ln() * 65536.0).round() as i64
                })
                .collect()
        })
        .collect()
}

/// `ShogiProgressKPAbs::bucket{,_board}` が受け付ける `num_buckets` の上限。
/// 返り値が `u8` のため `bucket = num_buckets - 1` が `u8::MAX = 255` に収まる
/// 256 までを許容する (LayerStack 既定 9 を含む全 production 用途に十分な margin)。
pub const MAX_NUM_BUCKETS: usize = u8::MAX as usize + 1;

#[cfg(test)]
mod tests {
    use super::*;

    /// shogi-format crate の `tests/data/sample.psv` (100 records × 40 bytes)。
    fn sample_psv_records() -> Vec<PackedSfenValue> {
        use std::path::PathBuf;
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../shogi-format/tests/data/sample.psv");
        let bytes =
            std::fs::read(&path).expect("sample.psv が読めない (../shogi-format/tests/data/)");
        assert_eq!(bytes.len() % 40, 0);
        assert_eq!(std::mem::size_of::<PackedSfenValue>(), 40);
        // SAFETY: `PackedSfenValue` は `#[repr(C)]` で `[u8; 40]` 1 個のみの POD
        // (不正ビットパターン無し、align = 1)。bytes 長は 40 の倍数。
        let recs: &[PackedSfenValue] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const PackedSfenValue, bytes.len() / 40)
        };
        recs.to_vec()
    }

    #[test]
    fn board_path_matches_legacy_path_on_real_psv() {
        // legacy delegating path (`for_each_active_index(&psv)` / `progress(&psv)` /
        // `bucket(&psv)` は内部で `psv.decode()` → board path) と board path
        // (`*_board(&psv.decode())`) が完全に一致する不変条件を実 PSV で確認する。
        //
        // NOTE: 重みは未ロード前提 (zero weights → progress = sigmoid(0) = 0.5、
        // bucket 4)。本 unit-test binary では `load_from_bin` を一切呼ばないので
        // `SHOGI_PROGRESS_KP_ABS_WEIGHTS` (OnceLock) は未 set のまま。
        let kpabs = ShogiProgressKPAbs;
        let records = sample_psv_records();
        let mut total_active = 0usize;
        for (i, psv) in records.iter().enumerate() {
            let board = psv.decode();

            let mut via_legacy = Vec::new();
            ShogiProgressKPAbs::for_each_active_index(psv, |idx| via_legacy.push(idx));
            let mut via_board = Vec::new();
            ShogiProgressKPAbs::for_each_active_index_board(&board, |idx| via_board.push(idx));
            assert_eq!(
                via_legacy, via_board,
                "record {i}: for_each_active_index の legacy/board path 不一致"
            );
            // 全 index が valid 範囲内。
            for &idx in &via_board {
                assert!(
                    idx < SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS,
                    "record {i}: idx {idx} 範囲外"
                );
            }
            total_active += via_board.len();

            assert_eq!(
                kpabs.progress(psv),
                kpabs.progress_board(&board),
                "record {i}: progress の legacy/board path 不一致"
            );
            assert_eq!(
                kpabs.bucket(psv, 9),
                kpabs.bucket_board(&board, 9),
                "record {i}: bucket の legacy/board path 不一致"
            );
            // zero weights → progress 厳密に 0.5。`floor(0.5 * N) = N/2` で
            // N=8 / N=9 ともに 4。
            assert_eq!(kpabs.progress(psv), 0.5, "record {i}: zero-weight progress");
            assert_eq!(
                kpabs.bucket(psv, 9),
                4,
                "record {i}: zero-weight bucket N=9"
            );
            assert_eq!(
                kpabs.bucket(psv, 8),
                4,
                "record {i}: zero-weight bucket N=8"
            );
        }
        assert!(
            total_active > 0,
            "sample.psv は実局面なので active index を持つ record があるはず"
        );
    }

    #[test]
    fn bucket_board_matches_bucket_zero_weights() {
        // 最小局面 (両玉のみ) でも legacy/board path が一致し、zero-weight で
        // progress = 0.5 / bucket = 4 (N=8 / N=9 共通) になることを確認。
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            ..Default::default()
        };
        board.black_king_sq = shogi_format::types::Square::new(4, 8);
        board.white_king_sq = shogi_format::types::Square::new(4, 0);
        let p = ShogiProgressKPAbs;
        assert_eq!(p.progress_board(&board), 0.5);
        assert_eq!(p.bucket_board(&board, 9), 4);
        assert_eq!(p.bucket_board(&board, 8), 4);
    }

    #[test]
    fn bucket_board_ranges_for_varied_n() {
        // zero weights → p = 0.5。N を 1..=9 まで振って `floor(p*N) = floor(0.5*N)`
        // が返ることを確認 (N が偶数なら N/2、奇数なら (N-1)/2、N=1 は常に 0)。
        let board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: shogi_format::types::Square::new(4, 8),
            white_king_sq: shogi_format::types::Square::new(4, 0),
            ..Default::default()
        };
        let p = ShogiProgressKPAbs;
        for n in 1..=9 {
            let want = ((0.5_f32 * n as f32).floor() as i32).clamp(0, n as i32 - 1) as u8;
            let got = p.bucket_board(&board, n);
            assert_eq!(got, want, "bucket_board mismatch for N={n}");
        }
    }

    #[test]
    fn q16_thresholds_match_inference_table() {
        // 推論側 (rshogi `progress_q16.rs` の THRESHOLDS) と
        // 同じ値。N = 8 / 9 / 16 を固定値で照合し、全 N で閾値の両側の割当を確認する。
        let expected: [(usize, &[i64]); 3] = [
            (8, &[-127527, -71999, -33477, 0, 33477, 71999, 127527]),
            (
                9,
                &[-136278, -82101, -45426, -14624, 14624, 45426, 82101, 136278],
            ),
            (
                16,
                &[
                    -177475, -127527, -96098, -71999, -51672, -33477, -16470, 0, 16470, 33477,
                    51672, 71999, 96098, 127527, 177475,
                ],
            ),
        ];
        let table = Q16_THRESHOLDS.get_or_init(build_q16_thresholds);
        for (n, want) in expected {
            assert_eq!(&*table[n], want, "N={n}");
        }
        for n in 1..=MAX_NUM_BUCKETS {
            assert_eq!(q16_sum_to_bucket(i64::MIN, n), 0);
            assert_eq!(q16_sum_to_bucket(i64::MAX, n) as usize, n - 1);
            for (i, &t) in table[n].iter().enumerate() {
                assert_eq!(q16_sum_to_bucket(t - 1, n) as usize, i, "N={n}");
                assert_eq!(q16_sum_to_bucket(t, n) as usize, i + 1, "N={n}");
            }
        }
    }

    #[test]
    fn q16_quantization_rounds_half_away_from_zero_and_saturates() {
        let got: Vec<i32> = [
            0.5 / 65536.0,
            -0.5 / 65536.0,
            1.49 / 65536.0,
            -1.49 / 65536.0,
            f64::MAX,
            -f64::MAX,
        ]
        .into_iter()
        .map(quantize_q16)
        .collect();
        assert_eq!(got, [1, -1, 1, -1, i32::MAX, i32::MIN]);
    }

    #[test]
    fn q16_zero_weights_match_f32_bucket() {
        let board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: shogi_format::types::Square::new(4, 8),
            white_king_sq: shogi_format::types::Square::new(4, 0),
            ..Default::default()
        };
        let p = ShogiProgressKPAbs;
        assert_eq!(p.progress_q16_sum_board(&board), 0);
        for n in 1..=16 {
            assert_eq!(
                p.bucket_q16_board(&board, n),
                p.bucket_board(&board, n),
                "N={n}"
            );
        }
    }

    #[test]
    fn load_rejects_non_finite_coefficients() {
        let mut bytes = vec![0u8; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * 8];
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            bytes[8..16].copy_from_slice(&value.to_le_bytes());
            let err = ShogiProgressKPAbs::load_from_bytes(&bytes).err().unwrap();
            assert!(err.contains("coefficient 1 is not finite"), "{err}");
        }
    }

    #[test]
    #[should_panic(expected = "num_buckets must be in [1, 256]")]
    fn bucket_board_rejects_too_large_num_buckets() {
        // 上限 (MAX_NUM_BUCKETS = u8::MAX + 1 = 256) 超えは panic で reject される
        // (`u8` 返り値で `bucket = num_buckets - 1` が表現可能な範囲のみ受け付ける)。
        let board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: shogi_format::types::Square::new(4, 8),
            white_king_sq: shogi_format::types::Square::new(4, 0),
            ..Default::default()
        };
        let _ = ShogiProgressKPAbs.bucket_board(&board, MAX_NUM_BUCKETS + 1);
    }
}
