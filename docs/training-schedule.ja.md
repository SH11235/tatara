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

`--lr-schedule one-cycle --lr-total-steps N` は通算の optimizer 更新回数で
進み、`--superbatches` とは独立です。N は正の整数で、他の schedule では指定できません。
initial は `--lr / --lr-div-factor`、final は initial を `--lr-final-div-factor`
で割った値です。warmup は浮動小数点の境界 `--lr-warmup-pct * N - 1` で終わり、
丸めません。0 始まりの更新番号を使い、PyTorch と同じ half-cosine の区間で、warmup は
その境界で、cooldown は `N - 1` で終わります。warmup の割合は f32 で保持し、境界の
計算では f64 に変換します。境界が整数でないと、max LR ちょうどの更新が無いことが
あります。warmup の割合が 1.0 なら、更新 `N - 1` で max LR に達します。更新 `N` 以降は
更新 `N - 1` の値を保持します。warmup 0 では cooldown から始め、更新 0 で長さ 0 の
warmup 区間になる場合は max LR を使います。

checkpoint に N と完了した更新回数を保存します。resume で N を省略すると保存値を
使い、異なる N や schedule は拒否します。更新回数の horizon を持たない checkpoint
から、この方式への切り替えも拒否します。batch と 1 superbatch あたりの更新数を
変えても、保存した更新回数と LR 曲線を引き継ぎます。その他の one-cycle option は
resume 時にも同じ値を指定してください。新規学習で `--lr-total-steps` を省略すれば
従来の superbatch 単位の計算です。

`--lr-warmup-steps` は `one-cycle`（自前の warmup を持つ）を除く任意の schedule を、
最初の superbatch 内の batch 単位 warmup で追加ラップする。

`--l1-lr-mult` は FT の直後にある全結合層の重みに学習率の倍率を掛ける。
LayerStacks の両 head で L1 の共有重みも対象になる。省略時は LayerStacks では
`--dense-lr-mult`、simple では 1.0 を使う。L1 の weight decay 係数は LayerStacks
では dense の値（共有 L1 の明示的な上書きを除く）、simple では全体の値に従う。resume 時はそのコマンドラインの
倍率を使い、レシピの比較にも含める。

`--l1-shared-weight-decay` は LayerStack の全 bucket 共通の L1 重みだけの decay を上書きする。両 head で利用でき、bucket ごとの L1 重み・他層・bias には影響しない。学習率は `--l1-lr-mult`（省略時は dense の倍率）のまま。値は有限かつ非負に限り、simple では拒否する。省略時は従来の dense decay を使う。

`--adam-bias-correction` は `--optimizer adamw` 専用で、`m_hat = m / (1 - beta1^t)`、`v_hat = v / (1 - beta2^t)` を使い、`w = w * (1 - lr * wd) - lr * m_hat / (sqrt(v_hat) + epsilon)` の後に既存の weight clamp を適用する。省略時は従来の補正なし AdamW。native CUDA の FT fused / 非 fused、cuda-oxide、FP16 optimizer state の各経路で同じ補正を行う。FP16 state の保存時の丸めは残る。共通 kernel に渡す step 係数とともに epsilon・decay 係数を調整し、実効 epsilon と decoupled decay の率を保つ。

resume 時もこれらの option を指定する。checkpoint は重み・moment・累積 optimizer step `t` を復元する。`--adam-bias-correction` は checkpoint に記録し、resume では同じ指定が要る (この項目の無い checkpoint は補正なしとみなす)。他の option は現在の CLI から決まる。明示した値は起動ログ・experiment の params・recipe fingerprint に記録し、省略時は項目を追加しない。

LayerStacks の学習では、`--ft-piece-lr-mult` が駒特徴の実の行（全 effect bucket を含む）に、
`--threat-lr-mult` が Threat 特徴の実の行（幅を縮めた行を含む）に適用される。Threat の倍率には
`--threat-profile` が必要。いずれも既定値は 1.0 で、schedule の学習率に
`--ft-lr-mult` を掛けた値にさらに掛ける。仮想行と PSQT の重みは FT group の
学習率を使う。weight decay の係数は FT group の値を使い、decoupled decay の強さは
各行の学習率に連動する。norm loss には schedule の学習率を使う。resume 時は
コマンドラインで倍率を指定する。

FT の重みに異なる学習率を使うには、native CUDA の融合 optimizer が必要。
この場合は PSQT と精度診断を無効にし、Threat の幅を縮める場合は 8 の倍数にする。

### resume 時の batch サイズ変更

`--batch-size` と `--batches-per-superbatch` の積を保ちながら変更できる。checkpoint と対応する `.loader.json` を一緒に保管すること。sampling の読みかけ chunk を含め、受理済み局面の位置から新しい batch サイズで再開する。教師の同一性と入力設定の照合は続ける。batch サイズ変更によって実際の shuffle 窓の局面数（`window_records`）が変わる場合は読み位置の復元を拒否する。窓の局面数を変えない batch サイズを選ぶか、`--resume-input restart` を明示して教師の先頭から読み直す。

`--lr-total-steps` を指定した one-cycle 以外は superbatch 番号を基準とし、保存済み horizon を復元するため batch 数の変更では曲線を伸縮しない。その他の schedule option は CLI 由来なので同じ値を指定する。optimizer の moment と step 数は引き継ぎ、小さい batch では 1 superbatch あたりの更新回数が増える。batch 単位の warmup と WDL jitter は引き続き batch 数を基準とするため、局面数で見た長さは変わる。

### Horizon と resume

**horizon**（curve が終端 LR に到達する superbatch）を持つのは
`linear` / `cosine` / `exponential` 減衰（その `--lr-final-superbatch`）と
`--lr-total-steps` を指定しない `one-cycle`。`--lr-final-superbatch` 省略時、horizon は
`--superbatches` に default するため、stateless な再構築だと `--superbatches` を
変えて resume するたびに curve が伸縮してしまう。

curve を再現可能に保つため、checkpoint は解決済 horizon を記録し `--resume` で
復元する。resume 時の horizon は次の優先順位で決まる:

1. 明示した `--lr-final-superbatch`（decay schedule）— 最優先。
2. checkpoint に保存された horizon。
3. `--superbatches` — fallback。

`--lr-total-steps` を指定しない one-cycle では、resume 時に保存した superbatch horizon が
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
