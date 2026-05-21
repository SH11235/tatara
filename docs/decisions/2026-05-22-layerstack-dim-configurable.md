# LayerStack 層次元の CLI 可変化

- **Status**: Accepted
- **Date**: 2026-05-22

## Context

`nnue_train` の LayerStack アーキは層次元 (`FT_OUT` / `L1_OUT` / `L2_OUT` /
`NUM_BUCKETS`) を compile-time `const` で固定していた。一方 Simple アーキは同じ
binary 内で層次元を CLI (`--arch` / `--l1` 等) から runtime で受け取る。学習実験
で FT 幅などを振るには LayerStack も runtime 可変化が要る。

LayerStack の host code は層次元を既に runtime 式 (`batch * FT_OUT` 等) で扱って
おり、`const` を runtime 値に差し替えれば足りる。可変化の難所は GPU kernel:
LayerStack の dense 層は **per-position の bucket 別重み行列** (9 bucket) を選択
する必要があり、Simple のような素の `cublasSgemm` では表現できない。このため
LayerStack は inline cuda-oxide `#[kernel]` の custom tiled matmul を持ち、その
一部が層次元を compile-time tile size として焼き込んでいる:

- `L1_OUT` / `L2_OUT` は `SharedArray<f32, N>` の `N` (cuda-oxide では const
  generic) と register accumulator 段数に依存する。
- `FT_OUT` は FT→L1 matmul の K 次元で、kernel 内では runtime loop bound として
  tile (16 幅) 単位に消化される。compile-time の array size には現れない。

## Decision

### 1. 可変化は次元ごとに段階導入する

`FT_OUT` を先に runtime 化し、`L1_OUT` / `L2_OUT` は後続で kernel を一般化して
から可変化する。`NUM_BUCKETS` は progress8kpabs bucketing と密結合 (4 kernel が
9-way の register unroll を持つ) のため可変化の対象外とする。

理由: host plumbing (層次元を `const` から runtime field へ移す) と GPU kernel の
tile 一般化は独立に検証できる。`FT_OUT` は kernel 側が無改造で runtime 値を受け
られるため、最小の変更で先行導入できる。次元ごとに「既定値で従来と bit-identical」
を個別に回帰検証でき、kernel 改造を伴う変更と混ざらない。

### 2. custom kernel を `cublasSgemm` で置換しない

LayerStack dense 層の bucket 別重み選択は素の GEMM では表現できず、また置換は
累算順序が変わって既定構成の数値が変わる。後述の bit-identical 要件を満たせない
ため、`L1_OUT` / `L2_OUT` の可変化は custom kernel の一般化 (tile を runtime tile
数の loop で回す、`SharedArray` は対応上限まで over-allocate) で行う。

### 3. 既定構成は従来と bit-identical

層次元の既定値は現行値 (`FT_OUT=1536` / `L1_OUT=16` / `L2_OUT=32` / 9 bucket)。
既定では buffer サイズ・kernel launch 形状・累算順序が従来と一致し、既存
checkpoint と resume 互換を保つ。

### 4. CLI は次元ごとの個別 flag

`LayerstackArgs` に `--ft-out` を追加する (後続で `--l1` / `--l2`)。Simple の
`--arch "<ft>x2-<l1>-<l2>"` preset 文字列ではなく個別 flag を採る。LayerStack の
可変次元は少数で、preset 文字列パーサを足す利得が小さい。experiment.json の
`architecture` 文字列 (`LayerStack-{ft_out}-16-32-9bucket`) が人間可読な構成
要約として既に機能する。

### 5. `FT_OUT` は 128 の倍数に制約する

backward の `gather_and_sum_per_feature` は grid の y 次元を `FT_OUT / 128` で
launch する (block 128 thread)。128 の倍数でないと末尾行の勾配が計算されない。
`--ft-out` は起動時に「`> 0` かつ 128 の倍数」を検証する。128 は forward 系
kernel が要求する 4 / 16 の倍数も包含する。

### 6. runtime 次元は `GpuWorkspace` の field で持つ

feature set 依存の `ft_in` / `max_active` が既に `GpuWorkspace` の runtime field
である。層次元も同じ方式で field 化し、`GpuTrainer` は `self.ws` 経由で参照する。

### 7. checkpoint は v4 topology header をそのまま使う

raw checkpoint format v4 は arch-kind 名 + count-prefixed な `u64` topology header
を既に持つ。runtime の層次元をそのまま topology に書き、resume / init-from 時に
構成と照合する。format version の bump は不要。

## Consequences

- `FT_OUT` 可変化は GPU kernel を改造しない。host の buffer 確保・kernel launch
  arg・checkpoint topology・experiment.json・architecture 文字列が runtime 値を
  参照する。
- `L1_OUT` / `L2_OUT` の可変化は custom kernel (`dense_mm_*_bucket_tiled_l1` /
  `_l2` / `_l3`、`bias_grad_bucket_shared_sorted` 等) の tile 一般化を要する別
  作業として残る。それまで両次元は runtime field に載るが起動時 check で現行値に
  固定される。
- `NUM_BUCKETS` 可変化は scope 外。必要になれば register unroll の解消を含む
  別 ADR で扱う。
- 既定構成での bit-identical と既存 checkpoint resume 互換は維持される。
