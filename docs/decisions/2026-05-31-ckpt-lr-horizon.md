# LR schedule horizon を resume checkpoint に保存して --superbatches 非依存にする

- **Status**: Accepted
- **Date**: 2026-05-31

## Context

LR scheduler は run ごとに `(CLI args, superbatch index)` の stateless 関数として
再構築される。horizon を持つ schedule — `linear` / `cosine` / `exponential` 減衰の
終端 (`--lr-final-superbatch` 未指定時) と `one-cycle` の total — はその horizon を
`--superbatches` から解決する (`build_lr_scheduler` の
`lr_final_superbatch.unwrap_or(superbatches)` と `OneCycleLR::new(..., superbatches)`)。

このため resume 時に `--superbatches` を変えると、同じ superbatch でも返す learning
rate が変わり schedule 曲線が伸縮する。曲線を本来の horizon に固定したまま続きを
学習する (= 別 run と同じ LR 軌跡を再現する) ことが、stateless 再構築モデルでは
できない。`step` / `constant` / `drop` は horizon を持たないため影響を受けない。

raw checkpoint format (`RNRC`) は version 1..=4 を段階的に拡張してきた
(feature set header / producer run id / arch-kind + topology)。本 ADR は schedule
horizon の固定をこの format 拡張で扱う。

## Decision

### 1. format version 5 に horizon の `u64` を 1 個だけ足す

`step_count` の直後に LR-schedule horizon の `u64` を書く。値は curve が終端 LR に
到達する superbatch (decay の `final_superbatch`、one-cycle の `total_superbatch`)。
`0` は「horizon 未記録」の sentinel で、horizon を持たない schedule で書かれる
(有効な horizon は常に `>= 1` なので曖昧さは無い)。

### 2. 保存するのは horizon だけ — schedule 種別や他パラメータは保存しない

曲線の再現には schedule 種別・`--lr`・`--lr-final`・(one-cycle なら) warmup_pct /
div factors も要るが、これらは `--superbatches` から独立で、resume 時に同じ CLI flag
を渡せば再現される。`--superbatches` に依存して暗黙に動くのは horizon だけなので、
それだけを固定すれば「`--superbatches` 非依存」の目的を満たす。schedule 種別まで
checkpoint に焼くと、resume 時に `--lr-schedule` を変える運用 (decay の途中で別
schedule に切替える等) と衝突し、優先順位の定義が増える。horizon は schedule 種別を
跨いで「終端 superbatch」という同一の意味を持つため、種別非依存の `u64` 1 個で足りる。

### 3. resume 時の horizon 優先順位

`build_lr_scheduler` は horizon を次の優先順位で解決する (`resolve_lr_horizon`):

1. **明示した CLI horizon flag** — decay の `--lr-final-superbatch`。resume か否かに
   関わらず最優先。意図的な schedule 再計画を許す。
2. **resume した保存 horizon (v5+)** — `--superbatches` 由来の default を上書きし、
   曲線を `--superbatches` から独立に再現する。
3. **`--superbatches` 由来の default** — 新規 run、または保存 horizon を持たない
   checkpoint (v1..=4 / horizon を持たない schedule) の fallback。

`one-cycle` は専用の horizon flag を持たない (horizon = `--superbatches`) ため、
resume では保存 horizon が常に `--superbatches` を上書きする。明示上書きは存在しない。

「保存 horizon を常に優先し、明示 flag を無視する」案 (厳密再現を強制) も取りうるが、
power user が resume 時に schedule を再計画する余地を残すため、明示 flag を最優先とする。
どちらの分岐に入ったかは operator 向けに 1 行 log を出す。

### 4. version 1..=4 は後方互換で従来どおり

`load_raw_checkpoint` は引き続き version 1..=4 を受理する。これらは horizon header を
持たないため `lr_horizon = None` と解釈し、horizon は CLI 値 (上記 §3 の 1 / 3) から
再構築する — 現行挙動を bit-identical に維持する。`step` の resume 挙動も不変 (horizon
を持たないため §3 のどの分岐でも CLI 値で再構築される)。

## Consequences

- horizon は trainer ではなく `LrScheduler::horizon()` から導出する。scheduler は
  既に `final_superbatch` / `total_superbatch` を field に持つので、save 時に
  `lr_scheduler.horizon()` を読むだけで済み、別経路で horizon を threading しない。
- `save_resume_checkpoint` (trait) / `save_raw_checkpoint` (両 trainer) は
  `lr_horizon: Option<usize>` を受け取り header に書く。`load_raw_checkpoint` は
  `(superbatch, producer_run_id, lr_horizon)` を返し、resume driver が
  `build_lr_scheduler(cli, resumed_horizon)` に渡す。
- experiment.json の `lr_schedule` (effective schedule の Display) は
  `build_lr_scheduler` の結果を文字列化したものなので、resume 時の解決済 horizon を
  自動的に反映する (追加の記録経路は不要)。
- ロールバック: v5 を書く revision を revert すれば writer は v4 を書くようになり、
  v5 checkpoint は `version > RAW_CKPT_VERSION` で reject される。v4 以前の
  checkpoint は新旧どちらの reader でも従来どおり読める。
