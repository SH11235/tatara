[English](training-schedule.md) | **日本語**

# 学習スケジュール

1 つの run には独立した 2 つの scheduler がある: **学習率**（`--lr-schedule`）と
**WDL lambda**（`--wdl` / `--start-wdl` / `--end-wdl`）。どちらも CLI 引数と
superbatch index の関数として run ごとに再計算される。学習手順そのものは
[docs/training-quickstart.ja.md](training-quickstart.ja.md) を、各フラグの正確な
構文・範囲・default は `nnue-train --help` を参照（フラグ単位の説明はヘルプ
テキストが唯一の真実源）。本ドキュメントはヘルプでは書けない、各 scheduler の
挙動と `--resume` との相互作用を扱う。

## 学習率スケジュール（`--lr-schedule`）

`--lr-schedule` は学習率を run 全体でどう動かすかを選ぶ。default の `step` は
従来の `StepLR` と bit-identical。選択肢:

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

## 値の記録先

実効スケジュールは run の `experiment.json` の `params` に記録される。`lr_schedule`
は解決済の LR schedule 文字列を持ち、実効 horizon も含む（resume した run では
`--superbatches` 由来の default ではなく復元された horizon が出る）。`wdl`
フィールドは常に存在し（一定 lambda の値）、線形 taper のときは追加で
`start_wdl` / `end_wdl` が記録される（taper でないときは省略）。taper 時は
scheduler が `lambda` を `start_wdl` / `end_wdl` から決めるため `wdl` の値は
使われない。`test_loss` は各 superbatch で `train_loss` と同じ `lambda` で
計算されるので、両者は同じスケールに乗る（[held-out validation](held-out-validation.ja.md)
の「指標の読み方」を参照）。
