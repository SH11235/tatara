# 重みクランプを per-layer 量子化由来値に migrate する

- **Status**: Accepted
- **Date**: 2026-05-31

## Context

Ranger optimizer は学習中に全 trainable tensor (FT weight/bias / L1 / L1f / L2 /
L3 / PSQT) を **一律 `±1.98`** で clip していた (`RangerParams::min_weight /
max_weight` の単一値を全 group の `radam_step` launch に同値で渡していた)。これは
1 値を全 layer に流用する暫定実装で、各 tensor が実際に量子化される dtype・scale
と整合していなかった:

- **FT weight/bias** は i16 (scale QA=127) で量子化され、飽和点は `±32767/127 ≈
  ±258`。`±1.98` clip は飽和点より 2 桁きつく、110M+ params の FT が育つ余地を
  constraint で潰していた。nnue-pytorch も training 中 FT を clip しない
  (`include_input=False`)。
- **L1/L1f/L2/L3 weight** は i8 (scale QB=64) で量子化され、飽和点は `±127/64 ≈
  ±1.984`。`±1.98` はその近似値で実害は無いが、量子化定数を変えても追従しない
  hard-coded 値だった。
- **L1/L2/L3 bias と PSQT** は i32 (scale QA·QB=8128) で量子化され、飽和点は
  `±2^31/8128 ≈ ±264k` と事実上 unbounded。`±1.98` clip は量子化上の根拠が無い
  tight な制約だった。

### L3 weight 式に関する訂正 (当初案からの変更)

当初の設計案は L3 (output) weight の clip を **loss kind 依存**の
`±127² / (nnue2score · QB)` (sigmoid ≈ ±0.87 / WRM ≈ ±0.42) とするものだった。
これは nnue-pytorch / Stockfish の量子化方式 — 出力層が専用 scale
(`weight_scale_out`) を持ち nnue2score を出力 weight scale に畳み込む — に由来する
式である。

**tatara はこの方式を採らない。** `save_quantised`
(`crates/nnue-format/src/{layerstack_weights,simple_weights}.rs`) は L3 weight を
他の dense weight と同じ `round(w · QB)` clamp i8、すなわち scale QB=64 で量子化
する。nnue2score は出力 weight ではなく推論側 `fv_scale` (= `round(QA·QB/scale)`、
arch 文字列に格納し engine が global divisor として割る) に畳み込まれる。したがって
tatara の L3 weight 飽和点は他の i8 dense weight と同一の `±127/QB` であり、
**loss kind に依存しない**。

出力層を nnue-pytorch 風の専用 scale に変えて量子化解像度を上げる案 (loss 依存式が
整合するのはこの場合) は、量子化フォーマット側に踏み込む別実験として分離する
(本 ADR の Decision には含めない、後述)。

## Decision

一律 `±1.98` を廃止し、各 tensor の clip を **量子化 dtype の飽和点から導出**する。
`RangerParams` から `min_weight`/`max_weight` field を削除し (uniform clamp mode を
残さない)、kernel launch ごとに対象 group の値を渡す。`bins/nnue_train/src/arch.rs`
に 2 つの定数を定義する:

| 定数 | 値 | 適用 group |
|---|---|---|
| `W_CLAMP_QUANT` | `±i8::MAX/QB` (= ±127/64) | L1 / L1f / L2 / L3 weight、L1 / L1f / L2 bias |
| `W_CLAMP_NONE` | `f32::MIN ..= f32::MAX` | FT weight/bias、L3 bias、PSQT |

`W_CLAMP_NONE` の sentinel は `radam_step` の clamp 分岐 (`p < min` / `p > max`)
を有限 weight に対して常に false にし、kernel signature を変えずに「clamp skip」を
表現する。

clip 範囲の根拠を tensor 別にまとめる (i8 weight は `round(w·QB)` を [-128, 127] に
量子化するため正側端点 127 を採り対称 clamp `±i8::MAX/QB` とする。負側は厳密には
-128/QB まで取れるが、nnue 系では対称 clamp が慣例):

| group | 量子化 | 飽和点 | clip |
|---|---|---|---|
| FT weight / bias | i16 @ QA (=127 or SCReLU 255) | ±32767/QA ≈ ±128〜258 | 無し |
| L1 / L1f / L2 weight | i8 @ QB | [-128, 127]/QB | ±127/QB |
| L1 / L1f / L2 bias | i32 @ QA·QB | ±264k | ±127/QB |
| L3 weight | i8 @ QB | [-128, 127]/QB | ±127/QB (loss 非依存) |
| L3 bias | i32 @ QA·QB | ±264k | 無し |
| PSQT | i32 @ QA·QB | ±264k | 無し |

L1 / L1f / L2 bias は i32 量子化で飽和点まで余裕があるが、weight と同じ `±127/QB` に
据えて挙動を変えない (近似値 `±1.98` を量子化定数で formalize した neutral な値)。

QB=64 (i8 dense weight 量子化スケール) と i8 端点 127 は LayerStack / Simple 両
format で共通 (`nnue_format::{layerstack_weights,simple_weights}`)。FT 量子化スケール
QA は活性化依存 (CReLU/Pairwise 127、SCReLU 255) だが、FT は clamp しないので本
clamp 値には影響しない。

## Consequences

- **bit-identical 互換は捨てる**。clip 範囲が変わるので学習軌跡は従来と一致しない。
  ただし clip 範囲は optimizer state ではない (checkpoint format `RNGR` は
  momentum/velocity/slow_params のみ) ため、**既存 raw checkpoint からの resume は
  従来どおり動く**。
- weight 系の clip は実質 `±1.98 → ±1.984` で neutral。**学習挙動が実際に変わるのは
  FT 開放 (i16、110M+ params) と L3 bias / PSQT の開放**であり、SPRT が測る主因は
  FT 開放効果になる。
- legacy `±1.98` 一律 mode を打てる flag は追加しない。再導入したい場合は別 Issue で
  「なぜ ±1.98 一律で良かったか」の根拠を ADR 化すること。

## 受け入れ検証 (SPRT)

新 default vs 旧 `±1.98` 一律を SPRT で比較し、neutral 以上 (= 有意な regression
無し) を確認する。+Elo なら理想だが、neutral でも本 bug fix として merge する
(legacy 状態の維持に正当化が無いため)。有意な regression が出た場合はどの layer の
開放が原因かを bisect し、改善まで merge を留保する。SPRT 計測は production GPU
(RTX 3080 Ti) を要するため本 PR スコープ外の残タスクとする。

## 関連 / 後続

- **出力量子化 alignment (別設計、後続作業)**: L3 weight を nnue-pytorch 風の細かい
  scale で量子化し fv_scale を再導出することで、小さい WRM 出力 weight の量子化
  解像度を上げる実験。利得は `l3_w` 分布次第で不確実だが、本 ADR の clip 変更とは
  独立に SPRT 可能 (既存 float checkpoint の再量子化で測れる)。本 ADR は training-time
  clamp のみを対象とし、量子化フォーマットは変更しない。
- fv_scale / 量子化スケールの consumer 契約は
  `docs/decisions/2026-05-20-simple-quantised-format-engine-consumer.md`。
