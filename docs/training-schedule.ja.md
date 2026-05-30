[English](training-schedule.md) | **日本語**

# 学習スケジュール

学習を通して **WDL lambda** をどうスケジュールするか。学習手順そのものは
[docs/training-quickstart.ja.md](training-quickstart.ja.md) を、各フラグの正確な
構文・範囲・default は `nnue-train --help` を参照（フラグ単位の説明はヘルプ
テキストが唯一の真実源）。

## WDL lambda が制御するもの

各局面には、net を学習させる対象が 2 つある:

- **教師 score**（centipawn の評価値を loss の sigmoid / win-rate 変換に通したもの）
- **対局結果**（WDL: `0.0` 負け / `0.5` 引き分け / `1.0` 勝ち）

loss はこの 2 つを 1 つのスカラー `lambda` で blend する:

```
target = lambda * (対局結果) + (1 - lambda) * (教師 score)
```

つまり `lambda = 0` は教師評価値のみ、`lambda = 1` は対局結果のみで学習する。
blend は素の sigmoid-MSE loss でも win-rate-model loss（`--win-rate-model`）でも
同一で、両者で違うのは式の教師 score 側だけ。

`lambda` の範囲は `[0.0, 1.0]`。

## 一定 lambda（`--wdl`）

`--wdl <value>` は `lambda` を学習全体で固定する。default は `0.0`（教師 score
のみで学習）。対局結果に常に一定の重みを混ぜたいときに上げる。

```bash
target/release/nnue-train --data <psv> --wdl 0.3 ... simple
```

## 線形 taper（`--start-wdl` / `--end-wdl`）

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

実効スケジュールは run の `experiment.json` の `params` に記録される: 一定なら
`wdl`、線形 taper なら `start_wdl` / `end_wdl`（未使用時は省略）。`test_loss` は
各 superbatch で `train_loss` と同じ `lambda` で計算されるので、両者は同じ
スケールに乗る（[quickstart](training-quickstart.ja.md) の「メトリクスの読み方」を
参照）。
