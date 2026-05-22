# LayerStack / progress のバケット数 (N) の可変化

- **Status**: Accepted
- **Date**: 2026-05-23

## Context

LayerStack アーキは出力 bucket 数を compile-time `const NUM_BUCKETS = 9` で固定
している。各局面はこの 9 個の per-bucket 重み行列 (L1 delta / L2 / L3) から 1 個を
選んで評価し、どの bucket を選ぶかは progress 予測値から決まる:
`ShogiProgressKPAbs::bucket()` が `floor(p × 8).clamp(0, 7)` で **0..7 の 8 値**を
返す。

この 8 と 9 はずれている。index 8 の重み枠は確保・保存・GPU 計算されるが
`bucket()` が一度も emit しないため、未学習の死重になっている。

最良の bucket 数は実験しないと分からない。N を振って学習・棋力比較するには、
progress の binning と LayerStack の bucket 数を **CLI から可変 (任意 N)** にする
必要がある。

前提として `progress.bin` (KP-abs ロジスティック回帰の重みベクトル) は連続値
`p ∈ [0, 1]` を出す予測器で、**バケットの概念を持たない**。progress 学習の loss は
連続値の二乗誤差で、bucket 数は学習時の診断ヒストグラムにしか現れない。したがって
同じ `progress.bin` を任意の N で使え、N は `nnue_train` 実行時に自由に決められる
(`progress.bin` の再学習は不要)。

層次元 (`FT_OUT` / `L1_OUT` / `L2_OUT`) の CLI 可変化を扱った先行 ADR
(`2026-05-22-layerstack-dim-configurable.md`) は `NUM_BUCKETS` を scope 外とし、
「register unroll の解消を含む別 ADR で扱う」とした。本 ADR がそれにあたる。

## Decision

### 1. 単一の N が progress binning と LayerStack bucket 数の両方を駆動する

`bucket()` を `floor(p × N).clamp(0, N-1)` に一般化し、`nnue_train` の
`--num-buckets N` で N を与える。progress の binning と LayerStack の bucket 数は
同一 N から駆動して構造的に一致させ、現状の 8/9 ずれを解消する。

本 ADR が採るこの統一設計のもとでは、**既定構成が従来と bit-identical という性質は
成立しない** (`bucket()` が `floor(p × N)` に変わり、既定 N の選択に依らず局面 →
bucket 割当が変化する)。先行 ADR は層次元の既定可変化で既定 bit-identical を保ったが、
本変更は plumbing ではなく挙動変更 (死 bucket の解消) そのものであり、その性質を
意図的に手放す。

代替として、`bucket()` 既定は `floor(p × 8)` (歴史と同一) に固定し
`--num-buckets` 指定時のみ `floor(p × N)` に切り替える「legacy compat default」案も
ありうる (既定構成は bit-identical を保てる)。本 Issue の動機である 8/9 ずれの解消を
既定で達成しない (sweep 利用者だけが死 bucket 解消の恩恵を受け、既定利用者は
misalignment を温存する) ため採らない。resume 互換は §8 の既定 N 選択で別途維持する。

### 2. N ≤ 9 は GPU kernel の改造を要さない

production path の per-bucket kernel は、bucket 数を既に runtime 引数
(`num_buckets: u32`) で受けている。weight-backward kernel が持つ register
accumulator (`a0..a8`) も、`if num_buc >= k` ガード付きの flush と
`else if buc == k` chain による accumulate の両方が、index ≥ `num_buckets` を silent
skip する。N < 9 を引数として渡せば、index ≥ N に対応する accumulator は flush
されず、また `bucket()` が index ≥ N を emit しないため accumulate もされない ──
**N ≤ 9 は正しく処理される**。

bucket 数 9 を compile-time に焼いているのは以下のみで、kernel ロジックではない:

- `bins/nnue_train/src/arch.rs` の `const NUM_BUCKETS = 9`
- `crates/nnue-format/src/layerstack_weights.rs` の `const NUM_BUCKETS = 9`
- `trainer_layerstack.rs` の `assert!(NUM_BUCKETS == 9)` / `debug_assert!` 3 箇所
- `trainer_common.rs:285` `padded_sort_batch` の `(NUM_BUCKETS + 1) * 16`
- `bucket_counts_dev` / `bucket_offsets_dev` / `bucket_write_ctr_dev` buffer
  (`NUM_BUCKETS + 1` 要素)
- `exclusive_scan_aligned` 呼出の `n` 引数 (kernel コメント中の
  `n ≤ NUM_BUCKETS + 1 = 10` も同様)
- engine 向け `.bin` の暗黙 9-bucket 書き出し (§4)

したがって **N ≤ 9 の可変化は host plumbing のみ**で達成できる: 上記の `const` /
buffer サイズ / launch 引数を runtime N で駆動し、static assert を除去する。

非 sorted の `dense_mm_fwd_bucket_tiled_l1` (共有メモリ `W_TILE: SharedArray<f32,
2304>` = 9 × 16 × 16 固定) と `dense_mm_bwd_weight_bucket_tiled_l1` (`a0..a8`
register fan-out) は内部に bucket 数 9 を焼いた構造を持つが、production 未使用で
`gpu_cpu_equivalence_tests` からのみ参照される。production の N ≤ 9 可変化には
影響しない。本 ADR ではこれらを現状維持とし、§3 の N > 9 解禁時 (kernel 整理) に
合わせて削除または sorted variant への移行を検討する。

### 3. N > 9 は weight-backward kernel の bucket 次元一般化を要する

L2 / L3 の per-bucket weight backward kernel
(`dense_mm_bwd_weight_bucket_tiled_l2` / `_l3`) は、1 thread が batch を 1 回 scan し
9 個の register accumulator (`a0..a8`) へ bucket 別に fan-out する構造で、上限が 9 に
固定されている。N > 9 にはこの 2 kernel を一般化する。

一般化方式は先行 ADR の template (固定 unroll → grid 次元へ展開) を bucket 軸に
適用する: **register fan-out を撤廃し、bucket を `blockIdx.z` の grid 軸 + 単一
accumulator にする**。各 block が 1 bucket だけ担当する。これは L1 weight backward
(`dense_mm_bwd_weight_bucket_tiled_l1_sorted`) が既に採る形で、register 数が 9 → 1 に
減るため occupancy 面の回帰リスクは無い。

- **L3** (`out_dim = 1`、kernel コスト極小): batch を sort せず `blockIdx.z` を
  増やす。各 bucket block が batch 全体を scan し自 bucket のみ accumulate する。
  per-(bucket, ii) cell の累算順序は b ∈ [0, batch) の昇順で現行 kernel と identical で、
  N = 9 で bit-exact 一致する。redundant batch read 増は L3 がそもそも極小 kernel の
  ため絶対量は無視できる。
- **L2**: 同方式を実装候補 (一次案) とし、既定構成で現行 kernel と A/B 計測する。
  redundant batch read 増で既定構成が回帰した場合は sorted layout 版に書き換える
  (自 bucket の sorted slice のみ scan し read 1×)。採用判定は実装時の kernel 計測
  で行い、per-kernel 戦術は先行 ADR §2 の計測則に従う。

### 4. net file は自己記述的にする — `.bin` header に bucket 数を明示する

engine 向け `.bin` は layerstack section を `for buc in 0..NUM_BUCKETS` で 9 回
書くだけで、bucket 数を表す field を持たない (engine 側が 9 を hardcode して読む)。
可変 N では engine が layerstack section を何個読むかを知る必要がある。

**`.bin` header の `arch_str` の直後 (`ft_hash` の直前) に `num_buckets: u32` field を
挿入し、`NNUE_VERSION` を bump する**。`load_quantised` は両 `NNUE_VERSION` を読み
分け、bump 前の `NNUE_VERSION` を **暗黙の 9-bucket** として処理する compat path を
提供する (本 ADR の決定事項として engine 側の責務に含める)。

代替案の評価:

- file size からの逆算: LEB128 圧縮された FT section を解読し終わるまで bucket
  section の境界が決まらないため、loader が N を一意に得るには header parse と
  LEB128 解読の両方の整合に依存する (typed field と比べて依存経路が長い)。
- `arch_str` に `NumBuckets=N` token を埋める: arch_str は variable-length なので
  field 追加に version bump は不要 (forward-compatible)。一方で arch_str は
  YaneuraOu / nnue-pytorch 系の AffineTransform chain を表現する慣習に従っており、
  bucket 多重化を describe する slot は無い。新規 token の追加は文字列パーサに依存し
  typed field より曖昧で、`network_hash` の不変式にも組み込みづらい。
- **明示 `u32` field + `NNUE_VERSION` bump**: typed・一意・loader 実装が単純。
  `NNUE_VERSION` bump によって field 非対応の engine が field 付き `.bin` を silent
  誤読することを防ぐ (`NNUE_VERSION` mismatch で reject する)。本 ADR はこれを採る。

`.bin` header layout 変更を含む `nnue_train` 側と engine 側はロールアウトを協調する。
`NNUE_VERSION` が異なる `.bin` 同士は load 時に version mismatch で reject されるため、
両 version が混在しても silent corruption は起きない。

### 5. bucket 数を USI オプションで engine に渡さない

bucket 数は「その net がどう学習されたか」の属性で、net architecture の identity に
属する (層次元・活性化関数等と同列)。対局時にユーザーが選ぶ設定ではない。USI
オプションにすると net file と option が食い違ったとき engine がエラーを出さずに
誤った bucket を選び続け、評価が静かに壊れて SPRT を無効化する。net file は
self-describing であるべきで、bucket 数は §4 のとおり file から読む。

### 6. 「空バケット方式」を採らない

LayerStack を常に 9-bucket のまま据え置き、N < 9 のとき余りを死重として放置する案
(host plumbing をほぼ要さない) は採らない。主要な理由は **N > 9 解禁時 (§3) に
bucket スロット数を増やす必要があり、固定 9-bucket の file 形式では収容できず、
結局 file 形式の作り直しが発生する**こと。最初から自己記述的な N-bucket file を
出力すれば、N > 9 拡張時に kernel 改造のみで file 形式は流用できる。

副次的に N < 9 では死 bucket 分の per-bucket 重み (L1 + L2 + L3 で ~25K params /
bucket、量子化後 ~数十 KB) を file に保持することになるが、FT が支配的なため file
サイズ・推論時 memory footprint への絶対影響は無視できる。空バケット方式の否決理由は
このサイズコストではなく、format の自己記述性と N > 9 拡張可能性にある。

### 7. checkpoint は既存の topology header をそのまま使う

raw checkpoint (v4) の topology header は層次元列
`[ft_out, l1_out, l2_out, num_buckets]` を既に持つ。runtime の N をそのまま書き、
resume / init-from 時に照合する。format version の bump は不要。

N ≠ 9 で学習した checkpoint を異なる N で resume すると topology mismatch で
reject される (先行 ADR §7 の resume 互換規約の通常動作)。N を変更して継続学習する
場合は `--num-buckets <N>` の明示が必要。

### 8. 既定 `--num-buckets` は 9 とする

既定値を 9 とする。既存の 9-bucket checkpoint および配布済 net の checkpoint
topology と `.bin` 形状がフラグ無しで一致し、buffer 形状の resume / load が摩擦無く
通ることを優先する。この結果、既定の `bucket()` は `floor(p × 9)` となり、局面 →
bucket 割当が `floor(p × 8)` 時代から変化する ── これは Context に挙げた死 bucket の
解消そのものであり、意図した挙動変更である。

代替として既定 8 (binning 式が歴史と一致するが、既存 9-bucket checkpoint の resume に
`--num-buckets 9` の明示が必要) も検討したが、配布済 net の buffer 形状ゼロ摩擦
resume を優先して 9 を採る。

**短期 eval 劣化リスク**: 既存の 9-bucket 配布 net は `floor(p × 8)` binning で学習
されており、**index 8 の重みは初期化値のまま active 化された経験が無い** (progress が
0..7 のみ emit する設計のため)。既定 `floor(p × 9)` 下で配布済 net を resume すると、
progress `p ∈ [8/9, 1]` の局面 (~11%) が index 8 に振られ、未学習の初期化値で評価
することになる。継続学習 (fine-tune) で index 8 の重みが学習されるまで eval 精度は
局所的に劣化しうる。短期 bench だけで判断せず、本番 superbatch 数の長期 run + SPRT
で安定性を確認する (先行プロジェクトの精度 flag 検証規範に従う)。

## Consequences

- N ≤ 9 の可変化は `nnue_train` の host plumbing のみで、GPU kernel を改造しない。
  `bucket()` の N 化、`--num-buckets`、`const NUM_BUCKETS` の runtime 値化、buffer /
  launch 引数 / `padded_sort_batch` / `exclusive_scan_aligned` の N 駆動、static
  assert 除去、`.bin` header への field 追加 + `NNUE_VERSION` bump から成る。
- N > 9 の可変化は L2 / L3 weight-backward kernel 2 個の bucket 次元一般化を要する。
  N ≤ 9 の実験で N > 9 を試す価値が見えてから着手する。
- 推論エンジン (rshogi) は `.bin` header の `num_buckets` を読み、layerstack 配列と
  progress → bucket binning を N 化する。bump 前の `NNUE_VERSION` は暗黙 9-bucket
  として compat path で読む (§4)。`.bin` header layout 変更を含む `nnue_train` 側と
  engine 側はロールアウトを協調する。
- `experiment.json` の `architecture` 文字列と `num_buckets` field は runtime N を
  記録する。nnue-lab の lineage schema は num_buckets を既に持つため自動追従し、
  schema 側の追加変更は要らない。
- `gpu_cpu_equivalence_tests` の `dense_mm_fwd_bucket_tiled_l1` / `_bwd_weight_*_l1`
  (非 sorted) は production 未使用で W_TILE 2304 / `a0..a8` 固定を残す。N ≤ 9 可変化
  期では現状維持 (production path に影響しない)、N > 9 解禁時 (§3) に合わせて削除
  または sorted variant への移行を検討する。
- rshogi engine 側に残る literal 8 / 9 を含む識別子 (`SHOGI_PROGRESS8_NUM_BUCKETS`、
  `progress8kpabs` モード名、`L1536x16x32` 等) は既存 net 互換のため保持する。N
  一般化に伴う命名整理は engine 側で別途検討する範囲とし、本 ADR では決定しない。
- 既定構成の bit-identical は本 ADR で放棄する (バケット数統一に伴う意図的な挙動
  変更)。既存 checkpoint / 配布 net の buffer 形状互換は既定 N = 9 で維持するが、
  index 8 が未学習である点に伴う短期 eval 劣化リスクがある (§8)。長期 run + SPRT
  での安定性確認を運用要件とする。
- `progress.bin` の形式・内容は不変で、N 変更時の再学習は不要。
- ロールバック: `nnue_train` 側を `NNUE_VERSION` bump を含まない revision に pin し、
  配布 net も bump を含まない `nnue_train` で生成された `.bin` に pin することで戻
  せる。`NNUE_VERSION` の異なる `.bin` 同士は load 時に互いに reject されるため、
  ロールバック中に両 version が混在しても silent corruption は起きない。
