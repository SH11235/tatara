[English](training-schedule.md) | **日本語**

# 学習スケジュール

1 つの run には独立した 2 つの scheduler がある: **学習率**（`--lr-schedule`）と
**WDL lambda**（`--wdl` / `--start-wdl` / `--end-wdl` /
`--wdl-cycle-delta`）。どちらも CLI 引数と
superbatch index の関数として run ごとに再計算される。学習手順そのものは
[docs/training-quickstart.ja.md](training-quickstart.ja.md) を、各フラグの正確な
構文・範囲・default は `nnue-train --help` を参照（フラグ単位の説明はヘルプ
テキストが唯一の真実源）。本ドキュメントはヘルプでは書けない、各 scheduler の
挙動と `--resume` との相互作用を扱う。

## 学習率スケジュール（`--lr-schedule`）

`--lr-schedule` は学習率を run 全体でどう動かすかを選ぶ。選択肢:

- `step`（default）— `--lr-step` superbatch ごとに `--lr-gamma` を乗じる。
- `constant` — `--lr` を run 全体で固定。
- `drop` — `--lr-step` superbatch 経過後に 1 度だけ `--lr-gamma` を乗じる。
- `linear` / `cosine` / `exponential` — `--lr` から `--lr-final` まで
  `--lr-final-superbatch` までに減衰し、以降は保持。
- `one-cycle` — horizon の最初の `--lr-warmup-pct` で warmup し、以降は
  cosine で anneal する。

`--lr-warmup-steps` は `one-cycle`（自前の warmup を持つ）を除く任意の schedule を、
最初の superbatch 内の batch 単位 warmup で追加ラップする。

### Horizon と resume

**horizon**（curve が終端 LR に到達する superbatch）を持つのは
`linear` / `cosine` / `exponential` 減衰（その `--lr-final-superbatch`）と
`one-cycle`（その total）。`--lr-final-superbatch` 省略時、horizon は
`--superbatches` に default するため、stateless な再構築だと `--superbatches` を
変えて resume するたびに curve が伸縮してしまう。

curve を再現可能に保つため、checkpoint は解決済 horizon を記録し `--resume` で
復元する。resume 時の horizon は次の優先順位で決まる:

1. 明示した `--lr-final-superbatch`（decay schedule）— 最優先。
2. checkpoint に保存された horizon。
3. `--superbatches` — fallback。

`one-cycle` は専用の horizon flag を持たないため、resume では保存 horizon が
`--superbatches` を常に上書きする。これが記録される前に書かれた checkpoint（や
horizon を持たない `step` / `constant` / `drop`）は horizon を持たず
`--superbatches` に fallback する。したがって `step` はそもそも horizon を持たない
ので、`--superbatches` に依らず resume 時に同じ curve を再現する。

## WDL lambda スケジュール

WDL lambda は net を学習させる対象を制御する。各局面には対象が 2 つある:

- **教師 score**（centipawn の評価値を loss の sigmoid / win-rate 変換に通したもの）
- **対局結果**（WDL: `0.0` 負け / `0.5` 引き分け / `1.0` 勝ち）

loss はこの 2 つを 1 つのスカラー `lambda` で blend する:

```
target = lambda * (対局結果) + (1 - lambda) * (教師 score)
```

つまり `lambda = 0` は教師評価値のみ、`lambda = 1` は対局結果のみで学習する。
blend は素の sigmoid-MSE loss でも win-rate-model loss（`--win-rate-model`）でも
同一で、両者で違うのは式の教師 score 側だけ。`lambda` の範囲は `[0.0, 1.0]`。

以下の `... simple` はフルコマンドの省略形で、simple トレーナでは常に必須の
`--win-rate-model` フラグ群を含む（[quickstart](training-quickstart.ja.md#例-1-halfkp-nnue-を学習-simple-アーキ) を参照）。

### 一定 lambda（`--wdl`）

`--wdl <value>` は `lambda` を学習全体で固定する。default は `0.0`（教師 score
のみで学習）。対局結果に常に一定の重みを混ぜたいときに上げる。

```bash
target/release/nnue-train --data <psv> --wdl 0.3 ... simple
```

### 線形 taper（`--start-wdl` / `--end-wdl`）

`--start-wdl <a> --end-wdl <b>` は `lambda` を学習を通して線形補間する: 最初の
superbatch で `a`、最後（`--superbatches`）で `b` になり、その間は superbatch
ごとに等間隔で動く。`--resume` 時は taper が再開地点の superbatch から継続する
（最初からやり直さない）。

- 2 つのフラグは必ず両方指定する。片方だけは error。
- `--wdl` とは排他（同時指定は parse 時に reject される）。

典型的な使い方は、序盤は評価値重視・終盤は結果重視にする curriculum
——序盤は密な教師 score で安定した信号を学び、徐々に疎な対局結果へ重みを移す:

```bash
target/release/nnue-train --data <psv> --start-wdl 0.0 --end-wdl 0.5 ... simple
```

この blend の線形スケジューリングは nnue-pytorch の
`start_lambda → end_lambda` taper
（[model/lambda_utils.py](https://github.com/official-stockfish/nnue-pytorch/blob/e215624/model/lambda_utils.py)）
に倣ったもの。

superbatch が 1 つだけの run（`--superbatches 1`）は補間する区間が無いため、
taper は `--start-wdl` に縮退する。

### cosine cycle（`--wdl-cycle-delta`）

`--wdl-cycle-delta <d>` は `--wdl` を base とし、half-cosine で
`base + d` まで warmup してから base まで cooldown する。warmup の長さは
`--wdl-cycle-warmup-pct`（既定 `0.25`）で指定する。`--wdl-schedule-superbatch`
を指定すると horizon を固定でき、省略時は `--superbatches` を使う。schedule は
superbatch index の関数なので、horizon の入力が変わらない限り resume 後も同じ
曲線を続ける。学習率と違い horizon は checkpoint に保存されないため、resume で
`--superbatches` を延長しうる run では `--wdl-schedule-superbatch` を明示して
曲線を固定すること。horizon が 1 superbatch だと cycle の区間が無く lambda は
base のまま。`--wdl-cycle-warmup-pct 1.0` では cooldown が無く、horizon で
base へ戻る。delta は負でもよいが、`--wdl` との和が `[0, 1]` に収まる必要がある。

### 学習時の jitter

学習時はスケジュールの WDL 重みに乱数を足して、局面ごとの重みを求められる。

```text
lambda_i = clamp(base + batch_jitter + sample_jitter_i, 0, 1)
state = D * state + batch_STD * sqrt(1 - D^2) * N(0, 1)
```

- `--wdl-jitter-sample STD` は局面ごとに独立な正規乱数を足す。
- `--wdl-jitter-batch STD` は batch 内で共通の状態を足す。定常分布の標準偏差は STD。
- `--wdl-jitter-batch-decay D` で相関を指定する。範囲は `[0, 1)`。0 なら batch ごとに独立し、0 より大きい場合は batch STD も正でなければならない。
- `--wdl-jitter-cycle` は両方の乱数項に cycle の cosine 係数を掛ける。`--wdl-cycle-delta` が必要。
- `--wdl-jitter-seed N` で乱数系列を指定する。

STD は有限かつ 0 以上。数値フラグの既定値はすべて 0 で、cycle による倍率は既定で無効。seed の指定や cycle による倍率の有効化には、どちらかの STD が正であることが必要。score のみを使う局面と `--wdl-ignore-draws` の対象は WDL 重み 0 のまま。

CPU 参照実装、cuda-oxide、native CUDA では整数ハッシュと一様乱数への変換が一致する。正規乱数には Box–Muller 法の対数と余弦の丸めによる微小な差が生じることがある。

学習時の式は Stockfish nnue-pytorch の lambda jitter に対応する。tatara の lambda は WDL の重みで、Stockfish の lambda は score の重み。Stockfish は評価時にも jitter を掛ける。tatara は run 間で validation loss を比較できるよう、validation と eval-only で jitter を掛けず、スケジュールまたは `--validation-wdl` の値を使う。

tatara は superbatch 単位で cycle の係数を決め、スケジュールの horizon に達すると係数を 0 にして jitter を消す。Stockfish の `lambda_cycle_jitter` は step 単位の係数を使い、`lambda_cycle_warmup_pct = 1.0` のときは horizon 以降も 1 のまま。

相関のある batch jitter を再開するには checkpoint と対応する `.loader.json` を一緒に保管する。`--resume-input restart` で教師の読み込みを先頭からやり直す場合も、checkpoint に対応する batch jitter の状態は復元する。状態が欠けているか、保存時の optimizer step が checkpoint と異なる場合は、警告を出して 0 から再開する。

### 引き分けと validation

`--wdl-ignore-draws` を指定すると WDL が `0.5` の局面は lambda=0 として教師
score の target を使う。他のラベルには選択した schedule を使う。
`--validation-wdl <value>` を指定すると held-out validation と eval-only の lambda を
固定でき、省略時は training schedule を使う。

`--tree-sample-eval-only` は PSV の padding byte（byte 39）の bit 1 が立つ局面を、
WDL スケジュールによらず score 項のみ（WDL 重み 0）で学習する。PSV 形式の
`--data` が必要。bit 0 の dual-label gate は独立して働く。マーク付き局面は
外部の `--test-data` と `--test-tail-positions` のどちらでも拒否する。padding の
bit 0・bit 1 以外が立つ record もエラーにする。
`--tree-sample-accept F` はこのフラグと併用し、全教師のマーク付き局面を
確率 F（0 以上 1 以下、既定値 1）で採用する。マーク付き局面には
`--keep-rate`、`--min-ply`、`--ply-keep-curve`、`--bucket-keep-rates`、
`--mix-keep-rate` を適用しない。

## 値の記録先

実効スケジュールは run の `experiment.json` の `params` に記録される。`lr_schedule`
は解決済の LR schedule 文字列を持ち、実効 horizon も含む（resume した run では
`--superbatches` 由来の default ではなく復元された horizon が出る）。`wdl`
フィールドは常に存在し（一定 lambda の値）、線形 taper のときは追加で
`start_wdl` / `end_wdl` が記録される（taper でないときは省略）。taper 時は
scheduler が `lambda` を `start_wdl` / `end_wdl` から決めるため `wdl` の値は
使われない。`--validation-wdl` 未指定なら `test_loss` は各 superbatch の jitter なしの
WDL スケジュールを使う（[held-out validation](held-out-validation.ja.md) を参照）。
cycle run では `wdl_cycle_delta` と `wdl_cycle_warmup_pct` も記録され、
`wdl_schedule_superbatch` は明示した場合のみ記録される。`validation_wdl` は
指定時のみ、`wdl_ignore_draws` と `tree_sample_eval_only` は true の場合のみ記録される。
どちらかの jitter の標準偏差が正なら、設定を `wdl_jitter` に記録する。
