[English](held-out-validation.md) | **日本語**

# held-out validation

過学習や数値発散 (NaN) を SPRT 自己対局を待たずに早期検知するには、held-out
validation を有効化する。validation 用に「勾配更新に一切使わない局面」を別途
保持し、毎 superbatch 末に forward-only パスで集計する。training log は
`test_loss` / `test_acc` を出力 (console 向けの短い field 名)、`experiment.json`
は同じ値を `test_loss` / `test_accuracy` として記録する。opt-in 機能で、学習
全体の流れは [docs/training-quickstart.ja.md](training-quickstart.ja.md) を参照。

## 関連する 3 つの flag

| flag | 役割 | 種類 |
|---|---|---|
| `--test-tail-positions <N>` | held-out の **source**: `--data` の末尾 N 局面 | source A |
| `--test-data <PATH>` | held-out の **source**: 別 PSV または HCPE ファイル | source B |
| `--test-positions <K>` | 選んだ source から毎 superbatch **評価する局面数** | evaluation size、両 source 共用 |

`--test-tail-positions` と `--test-data` は held-out source の選択肢で、排他
(どちらか 1 つ、または両方未指定 = held-out 無効)。`--test-positions` は
source 選択とは別軸のパラメータで、選ばれた source の先頭から K 局面
(満タン batch 切上げ) を毎 superbatch 末に評価する。

## どちらの source を選ぶか

- **`--test-tail-positions <N>` (推奨)**: `--data` 自身の末尾 N 局面を切り
  分ける。training は `[0, file_end - N * 40)`、validation は
  `[file_end - N * 40, file_end)` を読み、両者は byte range レベルで disjoint
  なので contamination は構造的に発生しない。教師ファイル 1 本で training と
  validation 両方をまかなえるので、別 file を用意して同期管理する手間が要らな
  い。唯一のコストは training pool が N 局面減ること。教師全件 ≫ N の典型ケース
  (例: 1e9 局面の教師に対し N = 1e6) では 0.1% 未満で実害なし
- **`--test-data <path>`**: validation 専用の別 PSV または HCPE ファイル。`.hcpe`
  拡張子は Apery / dlshogi の 38-byte format、それ以外は従来の 40-byte PSV として
  読み込む。holdout 集合が
  `--data` と独立して用意済 (異なる generator / 異なる時期の局面群) で、その
  独立性を保ちたい積極的な理由があるときに使う。ergonomic 上の理由だけで `--data`
  を 2 本に分割する利点は無い

## 使用例

末尾 100 万局面を holdout に切り分け、うち先頭 1 万局面を毎 superbatch で評価:

```bash
target/release/nnue-train \
  --data <path/to/shuffled-psv.bin> \
  --test-tail-positions 1000000 \
  --test-positions 10000 \
  --output checkpoints/<run-name> --net-id <run-name> \
  --superbatches <N> --threads <N> \
  layerstack --progress-coeff <path/to/progress.bin>
```

## 指標の読み方

`test_loss` は `train_loss` と同じ loss kernel (sigmoid-MSE または WRM) +
同じ WDL lambda blend で計算するため、両者は単位・スケールが揃い同 superbatch 内
で直接比較できる。この blend の設定方法（一定の `--wdl` か、線形の
`--start-wdl` / `--end-wdl` taper か）は
[学習スケジュール](training-schedule.ja.md) を参照。`test_loss − train_loss` の差が広がっていけば過学習の兆候、
`test_loss` が `train_loss` より早く異常値に飛ぶようなら NaN 発散の早期検知に
なる。

`test_acc` / `test_accuracy` はモデル出力の符号と実対局結果の一致率 (引き分け
は分母から除外)。scale 不変なので、loss スケールが異なる run / 設定の間でも直接
比較できる。

## 単発評価と threat ablation (診断)

上記 held-out の仕組みを再利用し、**学習済**ネットを再学習なしで診断する flag。
`layerstack` subcommand 専用。

| flag | 動作 |
|---|---|
| `--eval-only` | 重みを load (`--init-from` か `--resume`) し、held-out を 1 回評価して `test_loss` / `test_accuracy` を出して終了。学習ループに入らない。 |
| `--threat-ablate <spec>` | 評価前に threat 特徴 row の pair-class 部分集合を 0 化し、その結果の `test_loss` 増加分でその部分集合の寄与を測る。threat ネット + `--init-from` 限定。 |
| `--threat-norm-dump` | load した threat 特徴重みの pair-class 別 L2 ノルム分解を出して終了。評価なし・GPU 不要・`--init-from` だけでよい。 |

`--eval-only` は held-out source (`--test-tail-positions` か `--test-data`) と
`--test-positions >= 1` が要る。source が `--test-data` のときは `--data` は不要
(`--test-tail-positions` は末尾を `--data` から取るので必須)。`--init-from` ネットの
feature set (`--threat-profile` 含む) は load 対象のネットと一致させること。

`--threat-ablate <spec>` の spec: `all` / `slider-attacker` / `step-attacker` /
`bigslider-attacker` / `defense` (攻撃側と対象が同 side) / `attack` (逆 side) /
`same-class` (攻撃 class == 対象 class) / `random:<seed>:<dims>` (threat 列を指定
本数だけ無作為に 0 化する再現可能な null baseline。構造的 spec の校正用)。
`--eval-only` を付けないと 0 化後に**学習**へ進むので、寄与測定には `--eval-only`
と併用する。

```bash
# slider-attacker threat block の寄与: 有無での held-out loss を比較。
target/release/nnue-train --init-from <threat.bin> --eval-only \
  --data <psv> --test-tail-positions 1000000 --test-positions 100000 \
  layerstack --threat-profile <profile> --progress-coeff <progress.bin>

target/release/nnue-train --init-from <threat.bin> --eval-only --threat-ablate slider-attacker \
  --data <psv> --test-tail-positions 1000000 --test-positions 100000 \
  layerstack --threat-profile <profile> --progress-coeff <progress.bin>

# モデルが threat 容量をどこに張ったか (host-only、GPU 不要、held-out データ不要):
target/release/nnue-train --init-from <threat.bin> --threat-norm-dump \
  layerstack --threat-profile <profile>
```
